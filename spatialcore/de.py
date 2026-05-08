"""Differential-expression helpers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .utils import get_layer_or_x, optional_import, require_columns


def make_pseudobulk_replicates(
    adata: Any,
    group_cols: Sequence[str],
    layer: str = "counts",
    min_cells: int = 1,
) -> pd.DataFrame:
    """Aggregate counts by observation metadata columns."""
    obs = adata.obs.reset_index(names="obs_id")
    require_columns(obs, group_cols, "adata.obs")
    X = get_layer_or_x(adata, layer)
    genes = pd.Index(adata.var_names).astype(str)
    records = []
    for keys, idx in obs.groupby(list(group_cols), observed=True).groups.items():
        idx_array = np.asarray(list(idx), dtype=int)
        if len(idx_array) < min_cells:
            continue
        summed = X[idx_array].sum(axis=0)
        values = np.asarray(summed).ravel()
        if sparse.issparse(summed):
            values = np.asarray(summed.toarray()).ravel()
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(dict(zip(genes, values)))
        row["n_cells"] = len(idx_array)
        records.append(row)
    return pd.DataFrame.from_records(records)


def run_pydeseq2(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    design: str,
    contrast: Sequence[str] | None = None,
    n_cpus: int = 1,
):
    """Run PyDESeq2 on explicit count and metadata tables."""
    pydeseq2_dds = optional_import("pydeseq2.dds", "modeling")
    pydeseq2_ds = optional_import("pydeseq2.ds", "modeling")
    dds = pydeseq2_dds.DeseqDataSet(counts=counts, metadata=metadata, design=design, n_cpus=n_cpus)
    dds.deseq2()
    stats = pydeseq2_ds.DeseqStats(dds, contrast=contrast, n_cpus=n_cpus)
    stats.summary()
    return stats.results_df


def rank_genes_groups(
    adata: Any,
    groupby: str,
    method: str = "wilcoxon",
    layer: str | None = None,
    key_added: str = "rank_genes_groups",
    **kwargs: Any,
):
    """Run Scanpy rank genes groups and return a tidy result table."""
    sc = optional_import("scanpy", "core")
    out = adata.copy()
    if layer is not None:
        out.X = out.layers[layer].copy()
    sc.tl.rank_genes_groups(out, groupby=groupby, method=method, key_added=key_added, **kwargs)
    return out, format_de_results(out, key=key_added)


def format_de_results(adata: Any, key: str = "rank_genes_groups") -> pd.DataFrame:
    """Convert Scanpy rank genes groups results to a tidy DataFrame."""
    if key not in adata.uns:
        raise KeyError(f"`adata.uns['{key}']` was not found.")
    rg = adata.uns[key]
    records = []
    groups = rg["names"].dtype.names

    def field_value(field: str, group: str, index: int, default: float = np.nan):
        if field not in rg:
            return default
        values = rg[field]
        if getattr(values.dtype, "names", None) and group in values.dtype.names:
            return values[group][index]
        return default

    for group in groups:
        n = len(rg["names"][group])
        for i in range(n):
            records.append(
                {
                    "group": group,
                    "rank": i + 1,
                    "gene": rg["names"][group][i],
                    "score": field_value("scores", group, i),
                    "pval": field_value("pvals", group, i),
                    "pval_adj": field_value("pvals_adj", group, i),
                    "logfoldchange": field_value("logfoldchanges", group, i),
                }
            )
    return pd.DataFrame.from_records(records)
