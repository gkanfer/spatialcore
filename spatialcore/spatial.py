"""Spatial distance, proximity, neighborhood, and mixed-model helpers."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

from .utils import (
    get_layer_or_x,
    optional_import,
    require_columns,
    require_obsm,
    to_dense_array,
)


def compute_nearest_neighbor_distances(
    adata: Any,
    spatial_key: str = "spatial",
    n_neighbors: int = 2,
    obs_key: str = "nearest_neighbor_distance",
    inplace: bool = False,
):
    """Compute nearest-neighbor distances from spatial coordinates."""
    out = adata if inplace else adata.copy()
    coords = require_obsm(out, spatial_key)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    distances, indices = nn.fit(coords).kneighbors(coords)
    out.obsm[f"{obs_key}_indices"] = indices
    out.obsm[f"{obs_key}_distances"] = distances
    out.obs[obs_key] = distances[:, 1] if distances.shape[1] > 1 else distances[:, 0]
    return out


def compute_spatial_neighbors(
    adata: Any,
    spatial_key: str = "spatial",
    coord_type: str = "generic",
    n_neighs: int | None = None,
    radius: float | tuple[float, float] | None = None,
    delaunay: bool = False,
    library_key: str | None = None,
    key_added: str = "spatial",
    inplace: bool = False,
    **kwargs: Any,
):
    """Build a Squidpy spatial graph and store it in AnnData.

    Results follow Squidpy's convention:
    `adata.obsp[f"{key_added}_connectivities"]`,
    `adata.obsp[f"{key_added}_distances"]`, and `adata.uns[key_added]`.
    """
    sq = optional_import("squidpy", "spatial")
    out = adata if inplace else adata.copy()
    require_obsm(out, spatial_key)

    params: dict[str, Any] = {
        "spatial_key": spatial_key,
        "coord_type": coord_type,
        "delaunay": delaunay,
        "library_key": library_key,
        "key_added": key_added,
    }
    if n_neighs is not None:
        params["n_neighs"] = n_neighs
    if radius is not None:
        params["radius"] = radius
    params.update(kwargs)

    sq.gr.spatial_neighbors(out, **params)
    return out


def compute_spatial_autocorrelation(
    adata: Any,
    genes: Sequence[str] | str | None = None,
    mode: str = "moran",
    layer: str | None = None,
    connectivity_key: str = "spatial_connectivities",
    key_added: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Compute global spatial autocorrelation with Squidpy.

    `mode="moran"` computes Moran's I; `mode="geary"` computes Geary's C.
    The spatial graph should already exist, usually from
    `compute_spatial_neighbors(..., key_added="spatial")`.
    """
    if connectivity_key not in adata.obsp:
        raise KeyError(
            f"`adata.obsp['{connectivity_key}']` was not found. "
            "Run `compute_spatial_neighbors` first or provide an existing graph key."
        )
    sq = optional_import("squidpy", "spatial")

    result = sq.gr.spatial_autocorr(
        adata,
        mode=mode,
        genes=genes,
        layer=layer,
        connectivity_key=connectivity_key,
        copy=True,
        **kwargs,
    )
    if result is None:
        result_key = key_added or ("moranI" if mode == "moran" else "gearyC")
        if result_key not in adata.uns:
            raise RuntimeError("Squidpy did not return or store spatial autocorrelation results.")
        result = adata.uns[result_key]
    result = result.copy() if hasattr(result, "copy") else pd.DataFrame(result)
    if key_added is not None:
        adata.uns[key_added] = result.copy()
    return result


def compute_local_moran(
    adata: Any,
    genes: Sequence[str] | str,
    graph_name: str = "spatial",
    obsm_key: str = "local_moran",
    inplace: bool = False,
    **kwargs: Any,
):
    """Compute local Moran's I with Voyager when available.

    Voyager currently provides the local-Moran workflow used in the reference
    notebooks. If Voyager is unavailable, this function raises an optional
    dependency error with the Visium HD install hint.
    """
    vp = optional_import("voyagerpy", "visiumhd")
    local_moran = getattr(getattr(vp, "spatial", None), "local_moran", None)
    if local_moran is None:
        raise NotImplementedError(
            "`voyagerpy.spatial.local_moran` was not found in the installed Voyager package. "
            "Local Moran's I is planned as a first-class fallback in a future release."
        )

    out = adata if inplace else adata.copy()
    gene_list = [genes] if isinstance(genes, str) else list(genes)
    missing = [gene for gene in gene_list if gene not in out.var_names]
    if missing:
        raise KeyError(f"Genes were not found in `adata.var_names`: {missing}")

    for gene in gene_list:
        result = local_moran(out, gene, graph_name=graph_name, **kwargs)
        if result is not None and obsm_key not in out.obsm:
            out.obsm[obsm_key] = result
    if obsm_key not in out.obsm:
        raise RuntimeError(f"Voyager completed but `adata.obsm['{obsm_key}']` was not created.")
    return out


def compute_spatial_lag(
    adata: Any,
    feature: str,
    distances_key: str = "distances",
    output_key: str | None = None,
    inplace: bool = False,
):
    """Compute inverse-distance-weighted spatial lag for an observation feature."""
    out = adata if inplace else adata.copy()
    if distances_key not in out.obsp:
        raise KeyError(f"`adata.obsp['{distances_key}']` was not found.")
    if feature not in out.obs:
        raise KeyError(f"`adata.obs['{feature}']` was not found.")
    dist = out.obsp[distances_key].copy()
    if sparse.issparse(dist):
        dist.data = 1.0 / (dist.data + 1e-10)
        row_sum = np.asarray(dist.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        weights = sparse.diags(1 / row_sum).dot(dist)
        lagged = weights.dot(out.obs[feature].to_numpy())
    else:
        dist = np.asarray(dist, dtype=float)
        with np.errstate(divide="ignore"):
            weights = 1.0 / (dist + 1e-10)
        weights[~np.isfinite(weights)] = 0
        row_sum = weights.sum(axis=1)
        row_sum[row_sum == 0] = 1.0
        lagged = (weights / row_sum[:, None]).dot(out.obs[feature].to_numpy())
    out.obs[output_key or f"lagged_{feature}"] = np.asarray(lagged).ravel()
    return out


def compute_var_by_distance(
    adata: Any,
    genes: Sequence[str],
    anchor_key: str,
    group_key: str,
    distance_key: str,
    bins: Sequence[float],
    layer: str | None = None,
) -> pd.DataFrame:
    """Summarize mean expression by distance bin for notebook-style curves."""
    return var_by_distance_curve(
        adata=adata,
        genes=genes,
        anchor_key=anchor_key,
        group_key=group_key,
        distance_key=distance_key,
        bins=bins,
        layer=layer,
    )


def var_by_distance_curve(
    adata: Any,
    genes: Sequence[str],
    anchor_key: str,
    group_key: str,
    distance_key: str,
    bins: Sequence[float],
    layer: str | None = None,
) -> pd.DataFrame:
    """Summarize mean expression by group and distance bin."""
    if group_key not in adata.obs or distance_key not in adata.obs or anchor_key not in adata.obs:
        raise KeyError("Required observation columns were not found.")
    X = to_dense_array(get_layer_or_x(adata[:, list(genes)], layer=layer))
    df = pd.DataFrame(X, columns=list(genes), index=adata.obs_names)
    meta = adata.obs[[group_key, distance_key, anchor_key]].copy()
    meta["distance_bin"] = pd.cut(meta[distance_key], bins=bins, include_lowest=True)
    long = df.join(meta).melt(
        id_vars=[group_key, distance_key, anchor_key, "distance_bin"],
        var_name="gene",
        value_name="expression",
    )
    return (
        long.groupby([group_key, anchor_key, "distance_bin", "gene"], observed=True)["expression"]
        .mean()
        .reset_index()
    )


def compute_co_occurrence(
    adata: Any,
    cluster_key: str,
    interval: Sequence[float] | None = None,
    spatial_key: str = "spatial",
    source: str | None = None,
    key_added: str = "co_occurrence",
    as_tidy: bool = True,
    inplace: bool = False,
    **kwargs: Any,
):
    """Run Squidpy co-occurrence and optionally return a tidy long table.

    The tidy output mirrors the spatial-proximity notebooks with columns
    `source`, `partner`, `distance`, and `ratio`.
    """
    sq = optional_import("squidpy", "spatial")
    out = adata if inplace else adata.copy()
    require_obsm(out, spatial_key)
    if cluster_key not in out.obs:
        raise KeyError(f"`adata.obs['{cluster_key}']` was not found.")
    out.obs[cluster_key] = out.obs[cluster_key].astype("category")

    occ, distances = sq.gr.co_occurrence(
        out,
        cluster_key=cluster_key,
        interval=interval,
        spatial_key=spatial_key,
        copy=True,
        **kwargs,
    )
    out.uns[key_added] = {"occurrence": occ, "distance": distances, "cluster_key": cluster_key}
    if not as_tidy:
        return out if inplace else (occ, distances)

    cats = list(out.obs[cluster_key].cat.categories)
    occ_arr = np.asarray(occ)
    dist_arr = np.asarray(distances)
    records = []
    if source is not None and source not in cats:
        raise KeyError(f"`source={source}` was not found in `adata.obs['{cluster_key}']`.")
    source_indices = range(len(cats)) if source is None else [cats.index(source)]
    for src_idx in source_indices:
        src_name = cats[src_idx]
        block = np.asarray(occ_arr[src_idx])
        if block.ndim == 1:
            block = block.reshape(1, -1)
        distance_values = dist_arr
        if block.shape[-1] == len(dist_arr) - 1:
            distance_values = dist_arr[1:]
        elif block.shape[-1] != len(dist_arr):
            distance_values = np.arange(block.shape[-1])
        for partner_idx, partner in enumerate(cats[: block.shape[0]]):
            for distance, ratio in zip(distance_values, block[partner_idx], strict=False):
                records.append(
                    {
                        "source": src_name,
                        "partner": partner,
                        "distance": float(distance),
                        "ratio": float(ratio),
                    }
                )
    return pd.DataFrame.from_records(records)


def compute_ripley_delta(
    observed: pd.DataFrame,
    simulated: pd.DataFrame,
    group_cols: Sequence[str],
    stat_col: str = "stats",
) -> pd.DataFrame:
    """Compute observed minus mean simulated Ripley-style statistics."""
    require_columns(observed, [*group_cols, stat_col], "observed")
    require_columns(simulated, [*group_cols, stat_col], "simulated")
    sim_mean = (
        simulated.groupby(list(group_cols), observed=True)[stat_col]
        .mean()
        .rename("sim_mean")
        .reset_index()
    )
    out = observed.merge(sim_mean, on=list(group_cols), how="left")
    out["delta"] = out[stat_col] - out["sim_mean"]
    return out


def celltype_distance_table(
    adata: Any,
    target_mask: Sequence[bool] | pd.Series,
    cell_type_key: str,
    spatial_key: str = "spatial",
    sample_key: str | None = None,
    output_key: str = "distance_to_target",
) -> pd.DataFrame:
    """Compute distance from each cell to the nearest target cell."""
    coords = require_obsm(adata, spatial_key)
    target_mask = np.asarray(target_mask, dtype=bool)
    if not target_mask.any():
        raise ValueError("At least one target cell is required.")
    nn = NearestNeighbors(n_neighbors=1).fit(coords[target_mask])
    distances, _ = nn.kneighbors(coords)
    cols = [cell_type_key] + ([sample_key] if sample_key else [])
    records = adata.obs[cols].copy()
    records[output_key] = distances[:, 0]
    return records.reset_index(names="obs_id")


def run_celltype_distance_analysis(
    df: pd.DataFrame,
    distance_threshold: float = 60,
    cell_type_col: str = "cell_type",
    condition_col: str = "condition",
    timepoint_col: str = "timepoint",
    batch_col: str = "batch_pair",
    distance_col: str = "distance_to_target",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit cell-type-stratified distance models with batch random effects when possible."""
    require_columns(
        df,
        [cell_type_col, condition_col, timepoint_col, batch_col, distance_col],
        "df",
    )
    smf = __import__("statsmodels.formula.api", fromlist=["mixedlm", "ols"])
    data = df.copy()
    data["treatment"] = data[condition_col].astype(str) + "_" + data[timepoint_col].astype(str)
    data = data[data[distance_col] <= distance_threshold].copy()
    data["log_distance"] = np.log(data[distance_col] + 0.01)

    records = []
    diagnostics: dict[str, Any] = {"model_fits": {}, "skipped": []}
    for cell_type, df_ct in data.groupby(cell_type_col, observed=True):
        treatments = sorted(df_ct["treatment"].unique())
        if len(treatments) < 2:
            diagnostics["skipped"].append(
                {"cell_type": cell_type, "reason": "fewer_than_two_treatments"}
            )
            continue
        try:
            if df_ct[batch_col].nunique() > 1:
                model = smf.mixedlm(
                    "log_distance ~ C(treatment)",
                    data=df_ct,
                    groups=df_ct[batch_col],
                )
                fitted = model.fit(method="lbfgs", maxiter=200, reml=True, disp=False)
                model_notes = "MixedLM batch random intercept"
            else:
                fitted = smf.ols("log_distance ~ C(treatment)", data=df_ct).fit()
                model_notes = "OLS fallback"
            diagnostics["model_fits"][str(cell_type)] = fitted
        except Exception as exc:
            diagnostics["skipped"].append({"cell_type": cell_type, "reason": str(exc)})
            continue

        means = df_ct.groupby("treatment", observed=True)["log_distance"].mean()
        for trt1, trt2 in combinations(treatments, 2):
            vals1 = df_ct.loc[df_ct["treatment"] == trt1, "log_distance"]
            vals2 = df_ct.loc[df_ct["treatment"] == trt2, "log_distance"]
            stat = stats.ttest_ind(vals2, vals1, equal_var=False, nan_policy="omit")
            diff = means[trt2] - means[trt1]
            records.append(
                {
                    "cell_type": cell_type,
                    "contrast": f"{trt2} - {trt1}",
                    "n_cells_total": len(df_ct),
                    "n_cells_group1": len(vals1),
                    "n_cells_group2": len(vals2),
                    "n_batches": df_ct[batch_col].nunique(),
                    "effect_size_log": diff,
                    "effect_size_ratio": np.exp(diff),
                    "p_value": float(stat.pvalue) if pd.notna(stat.pvalue) else np.nan,
                    "model_notes": model_notes,
                }
            )
    results = pd.DataFrame.from_records(records)
    if not results.empty:
        _, qvals, _, _ = multipletests(results["p_value"].fillna(1), method="fdr_bh")
        results["q_value"] = qvals
    return results, diagnostics


def run_graphcompass_condition_comparison(
    adata: Any,
    library_key: str,
    cluster_key: str,
    condition_key: str,
    compute_spatial_graphs: bool = True,
    spatial_neighbors_kwargs: dict[str, Any] | None = None,
    run_wlkernel: bool = True,
    run_distance: bool = True,
) -> Any:
    """Run GraphCompass condition-comparison workflows on an AnnData object."""
    gc = optional_import("graphcompass", "niche")
    spatial_neighbors_kwargs = spatial_neighbors_kwargs or {
        "coord_type": "generic",
        "delaunay": True,
    }

    def _compare(func, **kwargs):
        try:
            return func(**kwargs)
        except TypeError as exc:
            if "condition_key" not in str(exc):
                raise
            kwargs.pop("condition_key", None)
            return func(**kwargs)

    if run_wlkernel:
        _compare(
            gc.tl.wlkernel.compare_conditions,
            adata=adata,
            library_key=library_key,
            cluster_key=cluster_key,
            condition_key=condition_key,
            compute_spatial_graphs=compute_spatial_graphs,
            kwargs_spatial_neighbors=spatial_neighbors_kwargs,
        )
    if run_distance:
        _compare(
            gc.tl.distance.compare_conditions,
            adata=adata,
            library_key=library_key,
            cluster_key=cluster_key,
            condition_key=condition_key,
            method="portrait",
            compute_spatial_graphs=False if run_wlkernel else compute_spatial_graphs,
            kwargs_spatial_neighbors=spatial_neighbors_kwargs,
        )
    return adata
