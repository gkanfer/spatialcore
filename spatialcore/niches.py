"""Spatial niche and similarity analysis helpers."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .utils import get_layer_or_x, optional_import, require_obsm


def run_cellcharter_grid(
    adata: Any,
    n_neighbors: Iterable[int],
    n_clusters: int = 3,
    use_rep: str = "X_pca",
    spatial_key: str = "spatial",
    sample_key: str | None = None,
    model_name: str = "gene",
    inplace: bool = False,
):
    """Run a CellCharter KNN grid and return the updated AnnData plus metrics."""
    sq = optional_import("squidpy", "spatial")
    cc = optional_import("cellcharter", "niche")
    out = adata if inplace else adata.copy()
    records = []
    for n in n_neighbors:
        graph_key = f"spatial_knn{n}"
        agg_key = f"X_cc_{model_name}_knn{n}_k{n_clusters}"
        label_key = f"niche_{model_name}_knn{n}_k{n_clusters}"
        sq.gr.spatial_neighbors(out, spatial_key=spatial_key, n_neighs=n, key_added=graph_key, library_key=sample_key)
        cc.gr.aggregate_neighbors(out, n_layers=1, use_rep=use_rep, out_key=agg_key, sample_key=sample_key)
        cluster = cc.tl.Cluster(n_clusters=n_clusters)
        cluster.fit(out, use_rep=agg_key)
        out.obs[label_key] = cluster.predict(out, use_rep=agg_key).astype(str)
        records.append(evaluate_niche_result(out, label_key=label_key, embedding_key=agg_key, conn_key=f"{graph_key}_connectivities"))
    return out, pd.DataFrame.from_records(records)


def evaluate_niche_result(
    adata: Any,
    label_key: str,
    embedding_key: str,
    conn_key: str,
    cell_type_key: str | None = None,
) -> dict[str, float | str]:
    """Evaluate niche separation, spatial coherence, and size balance."""
    labels = adata.obs[label_key].astype(str)
    emb = np.asarray(adata.obsm[embedding_key])
    vc = labels.value_counts(normalize=True)
    out: dict[str, float | str] = {
        "label_key": label_key,
        "embedding_key": embedding_key,
        "conn_key": conn_key,
        "n_clusters": int(labels.nunique()),
        "min_cluster_fraction": float(vc.min()),
        "max_cluster_fraction": float(vc.max()),
    }
    if labels.nunique() > 1 and emb.shape[0] > labels.nunique():
        out["silhouette"] = float(silhouette_score(emb, labels))
    else:
        out["silhouette"] = np.nan
    if conn_key in adata.obsp:
        conn = adata.obsp[conn_key]
        coo = conn.tocoo() if sparse.issparse(conn) else sparse.coo_matrix(conn)
        same = labels.iloc[coo.row].to_numpy() == labels.iloc[coo.col].to_numpy()
        out["spatial_same_niche_fraction"] = float(np.mean(same)) if len(same) else np.nan
    if cell_type_key is not None and cell_type_key in adata.obs:
        out["NMI_niche_vs_cell_type"] = float(normalized_mutual_info_score(labels, adata.obs[cell_type_key].astype(str)))
    return out


def niche_composition_table(
    adata: Any,
    niche_key: str,
    group_key: str,
    normalize: str = "niche",
) -> pd.DataFrame:
    """Create a niche composition table by another observation category."""
    if niche_key not in adata.obs or group_key not in adata.obs:
        raise KeyError("Requested observation columns were not found.")
    counts = pd.crosstab(adata.obs[niche_key], adata.obs[group_key])
    if normalize == "niche":
        return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    if normalize == "group":
        return counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    return counts


def cosine_similarity_centroids(
    adata: Any,
    query_groups: Sequence[str],
    reference_groups: Sequence[str],
    group_col: str = "cell_type",
    layer: str | None = None,
    scale_reference: bool = True,
) -> pd.DataFrame:
    """Compute cosine similarity between query and reference expression centroids."""
    if group_col not in adata.obs:
        raise KeyError(f"`adata.obs['{group_col}']` was not found.")
    groups = adata.obs[group_col].astype(str)
    X = get_layer_or_x(adata, layer)
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    ref_mask = groups.isin(reference_groups)
    query_mask = groups.isin(query_groups)
    X_ref = X[ref_mask]
    X_query = X[query_mask]
    ref_labels = groups[ref_mask].to_numpy()
    query_labels = groups[query_mask].to_numpy()
    if scale_reference:
        scaler = StandardScaler().fit(X_ref)
        X_ref = scaler.transform(X_ref)
        X_query = scaler.transform(X_query)
    ref_centroids = np.vstack([X_ref[ref_labels == group].mean(axis=0) for group in reference_groups])
    query_centroids = np.vstack([X_query[query_labels == group].mean(axis=0) for group in query_groups])
    return pd.DataFrame(cosine_similarity(query_centroids, ref_centroids), index=query_groups, columns=reference_groups)


def run_tacco_similarity(*args: Any, **kwargs: Any):
    """Run TACCO annotation/similarity workflows lazily."""
    tacco = optional_import("tacco", "niche")
    if hasattr(tacco, "tl") and hasattr(tacco.tl, "annotate"):
        return tacco.tl.annotate(*args, **kwargs)
    raise AttributeError("The installed `tacco` package does not expose `tl.annotate`.")
