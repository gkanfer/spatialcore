"""Preprocessing, normalization, clustering, and integration helpers."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .utils import optional_import


def normalize_log1p(
    adata: Any,
    target_sum: float = 1e4,
    counts_layer: str = "counts",
    log_layer: str = "log1p",
    inplace: bool = False,
):
    """Normalize total counts, log1p-transform, and preserve counts/log layers."""
    sc = optional_import("scanpy", "core")
    out = adata if inplace else adata.copy()
    if counts_layer not in out.layers:
        out.layers[counts_layer] = out.X.copy()
    sc.pp.normalize_total(out, target_sum=target_sum)
    sc.pp.log1p(out)
    out.layers[log_layer] = out.X.copy()
    return out


def select_hvgs(
    adata: Any,
    n_top_genes: int = 2000,
    layer: str = "counts",
    flavor: str = "seurat_v3",
    subset: bool = False,
    inplace: bool = False,
):
    """Select highly variable genes using Scanpy."""
    sc = optional_import("scanpy", "core")
    out = adata if inplace else adata.copy()
    sc.pp.highly_variable_genes(
        out,
        n_top_genes=n_top_genes,
        flavor=flavor,
        layer=layer if layer in out.layers else None,
        subset=subset,
    )
    return out


def run_pca_neighbors(
    adata: Any,
    n_comps: int = 50,
    n_neighbors: int = 30,
    use_rep: str | None = None,
    random_state: int = 0,
    inplace: bool = False,
):
    """Run PCA and nearest-neighbor graph construction."""
    sc = optional_import("scanpy", "core")
    out = adata if inplace else adata.copy()
    if use_rep is None:
        sc.pp.pca(out, n_comps=n_comps, random_state=random_state)
        use_rep = "X_pca"
    sc.pp.neighbors(out, n_neighbors=n_neighbors, n_pcs=n_comps, use_rep=use_rep)
    return out


def run_leiden_umap(
    adata: Any,
    resolution: float = 1.0,
    key_added: str = "leiden",
    random_state: int = 0,
    inplace: bool = False,
):
    """Run Leiden clustering and UMAP."""
    sc = optional_import("scanpy", "core")
    out = adata if inplace else adata.copy()
    sc.tl.leiden(out, resolution=resolution, key_added=key_added, random_state=random_state)
    sc.tl.umap(out, random_state=random_state)
    return out


def run_spatial_leiden(
    adata: Any,
    spatial_connectivities_key: str = "spatial_connectivities",
    key_added: str = "spatial_leiden",
    resolution: float = 1.0,
    inplace: bool = False,
    **kwargs: Any,
):
    """Run spatial Leiden on an existing spatial-connectivity graph.

    The expected graph is usually created with
    `spatialcore.spatial.compute_spatial_neighbors(..., key_added="spatial")`,
    which stores `adata.obsp["spatial_connectivities"]`.
    Extra keyword arguments are passed to the installed `spatialleiden` backend
    for notebook-style parameters such as `layer_ratio`, `directed`, or `seed`.
    """
    sl = optional_import("spatialleiden", "visiumhd")
    out = adata if inplace else adata.copy()
    if spatial_connectivities_key not in out.obsp:
        raise KeyError(f"`adata.obsp['{spatial_connectivities_key}']` was not found.")
    runner = getattr(getattr(sl, "tl", None), "spatialleiden", None) or getattr(
        sl,
        "spatialleiden",
        None,
    )
    if runner is None:
        raise AttributeError(
            "The installed `spatialleiden` package does not expose `spatialleiden`."
        )
    try:
        runner(
            out,
            adjacency=out.obsp[spatial_connectivities_key],
            resolution=resolution,
            key_added=key_added,
            **kwargs,
        )
    except TypeError as exc:
        if "adjacency" not in str(exc):
            raise
        if spatial_connectivities_key != "spatial_connectivities":
            out.obsp["spatial_connectivities"] = out.obsp[spatial_connectivities_key]
        runner(out, resolution=resolution, key_added=key_added, **kwargs)
    return out


def evaluate_clustering_grid(
    adata: Any,
    resolutions: Iterable[float],
    n_neighbors: Iterable[int] = (15, 30),
    n_pcs: int = 40,
    cluster_key_prefix: str = "leiden",
    random_state: int = 0,
) -> tuple[Any, pd.DataFrame]:
    """Evaluate a small grid of neighbors and Leiden resolutions."""
    sc = optional_import("scanpy", "core")
    out = adata.copy()
    records = []
    if "X_pca" not in out.obsm:
        sc.pp.pca(out, n_comps=n_pcs, random_state=random_state)
    for n in n_neighbors:
        sc.pp.neighbors(out, n_neighbors=n, n_pcs=n_pcs, key_added=f"{n}neig")
        for res in resolutions:
            key = f"{cluster_key_prefix}_{n}neig_res{str(res).replace('.', '_')}"
            sc.tl.leiden(
                out,
                resolution=res,
                key_added=key,
                neighbors_key=f"{n}neig",
                random_state=random_state,
            )
            records.append(
                {
                    "neighbors": n,
                    "resolution": res,
                    "cluster_key": key,
                    "n_clusters": out.obs[key].nunique(),
                }
            )
    return out, pd.DataFrame.from_records(records)


def voyager_transform(
    adata: Any,
    percentile_sum: float = 5,
    percentile_bin: float = 1,
    total_counts_key: str = "total_counts",
    n_genes_key: str = "n_genes_by_counts",
):
    """Compute notebook-style Voyager filter thresholds and masks."""
    out = adata.copy()
    sum_th = np.percentile(out.obs[total_counts_key], percentile_sum)
    gene_th = np.percentile(out.obs[n_genes_key], percentile_bin)
    out.obs["voyager_keep"] = (out.obs[total_counts_key] >= sum_th) & (
        out.obs[n_genes_key] >= gene_th
    )
    out.uns["voyager_thresholds"] = {
        "total_counts": float(sum_th),
        "n_genes_by_counts": float(gene_th),
    }
    return out
