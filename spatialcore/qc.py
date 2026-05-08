"""Quality-control helpers for AnnData and spatial omics workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .plotting import auto_figsize, publication_context
from .utils import matrix_nnz, matrix_sum, optional_import, require_obsm, save_figure_if_requested


def select_count_matrix(adata: Any, prefer: tuple[str, ...] = ("counts",)):
    """Select a count matrix from `adata.layers`, `adata.raw.X`, or `adata.X`."""
    for layer in prefer:
        if hasattr(adata, "layers") and layer in adata.layers:
            return adata.layers[layer], f"layers['{layer}']"
    if getattr(adata, "raw", None) is not None:
        return adata.raw.X, "raw.X"
    return adata.X, "X"


def compute_qc_metrics(
    adata: Any,
    matrix: Any | None = None,
    obs_name: str = "cell",
    column_names: dict[str, str | None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute per-observation count and detected-gene QC metrics."""
    if matrix is None:
        matrix, source = select_count_matrix(adata)
    else:
        source = "provided"

    column_names = column_names or {}
    id_col = column_names.get("id", f"{obs_name}_id")
    reads_col = column_names.get("reads", "n_reads")
    log_reads_col = column_names.get("log_reads", "log2_n_reads")
    genes_col = column_names.get("genes", "n_genes_detected")

    n_reads = matrix_sum(matrix, axis=1)
    n_genes = matrix_nnz(matrix, axis=1)

    qc_data: dict[str, Any] = {id_col: np.asarray(adata.obs_names)}
    summary_cols: list[str] = []
    if reads_col is not None:
        qc_data[reads_col] = n_reads
        summary_cols.append(reads_col)
    if log_reads_col is not None:
        qc_data[log_reads_col] = np.log2(n_reads + 1)
        summary_cols.append(log_reads_col)
    if genes_col is not None:
        qc_data[genes_col] = n_genes
        summary_cols.append(genes_col)

    qc_df = pd.DataFrame(qc_data)
    summary_df = summarize_qc_metrics(qc_df, summary_cols)
    metadata = {
        f"n_{obs_name}s": int(matrix.shape[0]),
        "n_genes": int(matrix.shape[1]),
        "matrix_used": source,
        "obs_type": obs_name,
    }
    return qc_df, summary_df, metadata


def summarize_qc_metrics(
    qc_df: pd.DataFrame,
    metric_cols: list[str] | None = None,
    percentiles: tuple[int, ...] = (5, 50, 95),
) -> pd.DataFrame:
    """Summarize QC metric columns with requested percentiles."""
    if metric_cols is None:
        metric_cols = [c for c in qc_df.columns if pd.api.types.is_numeric_dtype(qc_df[c])]
    records = []
    for col in metric_cols:
        values = pd.to_numeric(qc_df[col], errors="coerce").dropna()
        record: dict[str, Any] = {"metric": col, "mean": values.mean(), "std": values.std()}
        for q in percentiles:
            record[f"q{q:02d}"] = np.percentile(values, q) if len(values) else np.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def flag_mito_genes(
    adata: Any,
    prefix: str = "mt-",
    var_key: str = "MT",
    inplace: bool = True,
):
    """Flag mitochondrial genes and compute simple mitochondrial percentages."""
    out = adata if inplace else adata.copy()
    gene_names = pd.Index(out.var_names).astype(str)
    out.var[var_key] = gene_names.str.lower().str.startswith(prefix.lower())
    matrix, _ = select_count_matrix(out)
    total_counts = matrix_sum(matrix, axis=1)
    if out.var[var_key].any():
        mito_counts = matrix_sum(matrix[:, np.asarray(out.var[var_key])], axis=1)
    else:
        mito_counts = np.zeros(out.n_obs)
    out.obs["total_counts"] = total_counts
    out.obs["n_genes_by_counts"] = matrix_nnz(matrix, axis=1)
    out.obs[f"pct_counts_{var_key}"] = np.divide(
        mito_counts * 100,
        total_counts,
        out=np.zeros_like(total_counts, dtype=float),
        where=total_counts > 0,
    )
    return out


def filter_cells_genes(
    adata: Any,
    min_counts: int | None = None,
    min_genes: int | None = None,
    min_cells: int | None = None,
    inplace: bool = False,
):
    """Filter cells and genes using Scanpy while preserving a counts layer when possible."""
    sc = optional_import("scanpy", "core")
    out = adata if inplace else adata.copy()
    if "counts" not in out.layers:
        out.layers["counts"] = out.X.copy()
    if min_counts is not None:
        sc.pp.filter_cells(out, min_counts=min_counts)
    if min_genes is not None:
        sc.pp.filter_cells(out, min_genes=min_genes)
    if min_cells is not None:
        sc.pp.filter_genes(out, min_cells=min_cells)
    return out


def plot_qc_histograms(
    adata: Any,
    metrics: tuple[str, ...] = ("total_counts", "n_genes_by_counts"),
    bins: str | int = "auto",
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
):
    """Plot QC histograms and return figure and axes."""
    with publication_context(dpi=dpi):
        fig, axes = plt.subplots(1, len(metrics), figsize=auto_figsize(len(metrics), ncols=len(metrics)), dpi=dpi)
        axes = np.atleast_1d(axes)
        for ax, metric in zip(axes, metrics):
            if metric not in adata.obs:
                raise KeyError(f"`adata.obs['{metric}']` was not found.")
            sns.histplot(adata.obs[metric], bins=bins, kde=True, ax=ax, color="#4c78a8")
            ax.set_xlabel(metric)
            ax.set_ylabel("Count")
        fig.tight_layout()
        save_figure_if_requested(fig, save_path=save_path, dpi=dpi)
        if show:
            plt.show()
        return fig, axes


def plot_qc_boxplots(
    adata: Any,
    sample_col: str,
    metrics: tuple[str, ...] = ("total_counts", "n_genes_by_counts"),
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
):
    """Plot per-sample QC boxplots and return figure and axes."""
    if sample_col not in adata.obs:
        raise KeyError(f"`adata.obs['{sample_col}']` was not found.")
    with publication_context(dpi=dpi):
        fig, axes = plt.subplots(1, len(metrics), figsize=auto_figsize(len(metrics), ncols=len(metrics)), dpi=dpi)
        axes = np.atleast_1d(axes)
        obs = adata.obs.copy()
        for ax, metric in zip(axes, metrics):
            if metric not in obs:
                raise KeyError(f"`adata.obs['{metric}']` was not found.")
            sns.boxplot(data=obs, x=sample_col, y=metric, ax=ax, showfliers=False, color="#86bc86")
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel(sample_col)
            ax.set_ylabel(metric)
        fig.tight_layout()
        save_figure_if_requested(fig, save_path=save_path, dpi=dpi)
        if show:
            plt.show()
        return fig, axes


def plot_qc_spatial(
    adata: Any,
    color: str = "total_counts",
    spatial_key: str = "spatial",
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
):
    """Plot a QC metric on spatial coordinates."""
    from .plotting import plot_spatial_scatter

    coords = require_obsm(adata, spatial_key)
    if color not in adata.obs:
        raise KeyError(f"`adata.obs['{color}']` was not found.")
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], color: adata.obs[color].to_numpy()})
    return plot_spatial_scatter(
        df,
        color=color,
        categorical=False,
        size=2,
        title=color,
        save_path=save_path,
        dpi=dpi,
        show=show,
    )
