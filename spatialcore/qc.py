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


# ---------------------------------------------------------------------------
# MAD-based outlier detection (per-batch)
# ---------------------------------------------------------------------------


def is_outlier(adata: Any, metric: str, nmads: float = 5.0) -> np.ndarray:
    """Flag cells as outliers if they lie more than *nmads* MADs from the median.

    Parameters
    ----------
    adata : AnnData
        Must have `metric` in `.obs`.
    metric : str
        Column name in `adata.obs` to evaluate.
    nmads : float
        Number of median absolute deviations for the threshold.

    Returns
    -------
    np.ndarray of bool
        True for outlier cells.
    """
    from scipy.stats import median_abs_deviation

    M = adata.obs[metric].to_numpy(dtype=float)
    med = np.median(M)
    mad = median_abs_deviation(M)
    return (M < med - nmads * mad) | (M > med + nmads * mad)


def flag_outliers(
    adata: Any,
    metrics: tuple[str, ...] = ("log1p_total_counts", "log1p_n_genes_by_counts"),
    nmads: float = 5.0,
    batch_key: str | None = None,
    outlier_col: str = "outlier",
    inplace: bool = False,
) -> Any:
    """Flag outlier cells using MAD-based thresholds, optionally per batch.

    Parameters
    ----------
    adata : AnnData
    metrics : tuple of str
        obs columns to evaluate (log-scale metrics recommended).
    nmads : float
        MAD threshold.
    batch_key : str or None
        If provided, outlier detection is performed per batch.
    outlier_col : str
        Name of the boolean column added to `adata.obs`.
    inplace : bool
        If True, modify in place; otherwise return a copy.

    Returns
    -------
    AnnData with `outlier_col` in `.obs`.
    """
    out = adata if inplace else adata.copy()
    out.obs[outlier_col] = False

    if batch_key is not None and batch_key in out.obs.columns:
        for batch in out.obs[batch_key].unique():
            mask = out.obs[batch_key] == batch
            batch_adata = out[mask]
            batch_outlier = np.zeros(mask.sum(), dtype=bool)
            for metric in metrics:
                if metric in batch_adata.obs.columns:
                    batch_outlier |= is_outlier(batch_adata, metric, nmads)
            out.obs.loc[mask, outlier_col] = batch_outlier
    else:
        outlier_mask = np.zeros(out.n_obs, dtype=bool)
        for metric in metrics:
            if metric in out.obs.columns:
                outlier_mask |= is_outlier(out, metric, nmads)
        out.obs[outlier_col] = outlier_mask

    return out


# ---------------------------------------------------------------------------
# Plot: QC violin plots (scanpy-style)
# ---------------------------------------------------------------------------


def plot_qc_violin(
    adata: Any,
    keys: tuple[str, ...] = ("n_genes_by_counts", "total_counts", "pct_counts_MT"),
    groupby: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
):
    """Plot violin plots for QC metrics using Scanpy."""
    sc = optional_import("scanpy", "core")
    fig = sc.pl.violin(
        adata,
        keys=list(keys),
        groupby=groupby,
        jitter=0.4,
        multi_panel=True,
        show=False,
    )
    # sc.pl.violin returns None but draws on current figure
    current_fig = plt.gcf()
    current_fig.set_dpi(dpi)
    save_figure_if_requested(current_fig, save_path=save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(current_fig)
    return current_fig


# ---------------------------------------------------------------------------
# Plot: scatter of QC metrics (total_counts vs n_genes, colored by mito %)
# ---------------------------------------------------------------------------


def plot_qc_scatter(
    adata: Any,
    x: str = "total_counts",
    y: str = "n_genes_by_counts",
    color: str = "pct_counts_MT",
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
):
    """Scatter plot of two QC metrics colored by a third (e.g. mito %)."""
    with publication_context(dpi=dpi):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3.5), dpi=dpi)
        if color in adata.obs.columns:
            sc_plot = ax.scatter(
                adata.obs[x], adata.obs[y],
                c=adata.obs[color], cmap="RdYlGn_r",
                s=1, alpha=0.6, rasterized=True,
            )
            plt.colorbar(sc_plot, ax=ax, label=color)
        else:
            ax.scatter(
                adata.obs[x], adata.obs[y],
                s=1, alpha=0.6, color="#4c78a8", rasterized=True,
            )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x}")
        fig.tight_layout()
        save_figure_if_requested(fig, save_path=save_path, dpi=dpi)
        if show:
            plt.show()
        return fig, ax


# ---------------------------------------------------------------------------
# Plot: HVG mean-variance
# ---------------------------------------------------------------------------


def plot_hvg(
    adata: Any,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
):
    """Plot highly variable genes (mean vs. normalized variance)."""
    if "highly_variable" not in adata.var.columns:
        raise KeyError("Run HVG selection first (`sc.pp.highly_variable_genes`).")

    with publication_context(dpi=dpi):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=dpi)
        hvg = adata.var["highly_variable"]
        means = adata.var.get("means", pd.Series(dtype=float))
        dispersions = adata.var.get(
            "dispersions_norm",
            adata.var.get("variances_norm", pd.Series(dtype=float)),
        )
        if means.empty or dispersions.empty:
            # fallback to scanpy's built-in plot
            sc = optional_import("scanpy", "core")
            sc.pl.highly_variable_genes(adata, show=False)
            current_fig = plt.gcf()
            save_figure_if_requested(current_fig, save_path=save_path, dpi=dpi)
            if show:
                plt.show()
            return current_fig, plt.gca()

        ax.scatter(
            means[~hvg], dispersions[~hvg],
            s=1, alpha=0.4, color="#BBBBBB", label="Other", rasterized=True,
        )
        ax.scatter(
            means[hvg], dispersions[hvg],
            s=1, alpha=0.6, color="#E45756", label="Highly variable", rasterized=True,
        )
        ax.set_xlabel("Mean expression")
        ax.set_ylabel("Normalized dispersion")
        ax.legend(markerscale=5, frameon=False)
        fig.tight_layout()
        save_figure_if_requested(fig, save_path=save_path, dpi=dpi)
        if show:
            plt.show()
        return fig, ax


# ---------------------------------------------------------------------------
# Main QC workflow function
# ---------------------------------------------------------------------------


def run_qc_workflow(
    adata: Any,
    *,
    # Filtering parameters
    min_genes: int = 200,
    min_cells: int = 3,
    min_counts: int = 10,
    max_counts: int | None = None,
    max_genes: int | None = None,
    mito_prefix: str = "mt-",
    max_pct_mito: float | None = 20.0,
    # MAD-based outlier filtering
    use_mad_filtering: bool = False,
    mad_nmads: float = 5.0,
    mad_metrics: tuple[str, ...] = ("log1p_total_counts", "log1p_n_genes_by_counts"),
    batch_key: str | None = None,
    # Normalization
    normalize: bool = True,
    target_sum: float = 1e4,
    log_transform: bool = True,
    # HVG selection
    select_hvg: bool = True,
    n_top_genes: int = 2000,
    hvg_flavor: str = "seurat_v3",
    subset_hvg: bool = False,
    # Plotting
    plot: bool = True,
    show_plots: bool = True,
    save_dir: str | Path | None = None,
    dpi: int = 300,
    # Misc
    copy: bool = True,
) -> Any:
    """Run a complete QC workflow: filtering, normalization, HVG selection, and plotting.

    This function reproduces the QC pipeline used in the Xenium spatial
    transcriptomics preprocessing notebooks. It accepts an AnnData object and
    returns a processed copy (or modifies in place if ``copy=False``).

    Parameters
    ----------
    adata : AnnData
        Input annotated data matrix (cells x genes). Should contain raw counts.
    min_genes : int
        Minimum number of genes expressed per cell.
    min_cells : int
        Minimum number of cells expressing a gene.
    min_counts : int
        Minimum total UMI counts per cell.
    max_counts : int or None
        Maximum total counts per cell (None = no upper limit).
    max_genes : int or None
        Maximum genes detected per cell (None = no upper limit).
    mito_prefix : str
        Prefix for mitochondrial gene names (case-insensitive).
    max_pct_mito : float or None
        Maximum mitochondrial percentage. Cells above this are removed.
        Set to None to skip mitochondrial filtering.
    use_mad_filtering : bool
        If True, apply MAD-based outlier detection before hard-threshold filtering.
    mad_nmads : float
        Number of MADs for outlier detection.
    mad_metrics : tuple of str
        obs columns used for MAD outlier detection.
    batch_key : str or None
        If provided, MAD filtering is applied per batch.
    normalize : bool
        Whether to normalize total counts per cell.
    target_sum : float
        Target sum for normalization.
    log_transform : bool
        Whether to apply log1p transformation.
    select_hvg : bool
        Whether to select highly variable genes.
    n_top_genes : int
        Number of top HVGs to select.
    hvg_flavor : str
        Flavor for HVG selection ('seurat_v3', 'seurat', 'cell_ranger').
    subset_hvg : bool
        If True, subset adata to only HVGs. If False, mark in `adata.var`.
    plot : bool
        Whether to generate QC plots.
    show_plots : bool
        Whether to display plots (if False, figures are returned but not shown).
    save_dir : str or Path or None
        Directory to save plots. If None, plots are not saved.
    dpi : int
        DPI for saved figures.
    copy : bool
        If True, return a copy; if False, modify in place.

    Returns
    -------
    AnnData
        Processed AnnData object with:
        - ``.layers['counts']``: raw count matrix
        - ``.layers['log1p']``: log-normalized matrix (if normalize=True)
        - ``.var['highly_variable']``: HVG annotation (if select_hvg=True)
        - QC metrics in ``.obs``

    Examples
    --------
    >>> import scanpy as sc
    >>> import spatialcore
    >>> adata = sc.read_h5ad("my_data.h5ad")
    >>> adata_qc = spatialcore.qc.run_qc_workflow(
    ...     adata,
    ...     min_genes=200,
    ...     min_cells=3,
    ...     max_pct_mito=20.0,
    ...     n_top_genes=2000,
    ... )
    """
    sc = optional_import("scanpy", "core")

    out = adata.copy() if copy else adata

    # --- 1. Preserve raw counts ---
    if "counts" not in out.layers:
        out.layers["counts"] = out.X.copy()

    # --- 2. Compute QC metrics ---
    # Flag mitochondrial genes
    gene_names = pd.Index(out.var_names).astype(str)
    out.var["MT"] = gene_names.str.lower().str.startswith(mito_prefix.lower())

    sc.pp.calculate_qc_metrics(
        out,
        qc_vars=["MT"],
        percent_top=None,
        log1p=True,
        inplace=True,
    )

    # --- 3. MAD-based outlier filtering (optional) ---
    if use_mad_filtering:
        out = flag_outliers(
            out,
            metrics=mad_metrics,
            nmads=mad_nmads,
            batch_key=batch_key,
            outlier_col="outlier",
            inplace=True,
        )
        n_before = out.n_obs
        out = out[~out.obs["outlier"]].copy()
        n_removed = n_before - out.n_obs
        print(f"MAD filtering: removed {n_removed} outlier cells "
              f"({n_removed / n_before * 100:.1f}%)")

    # --- 4. Hard-threshold cell/gene filtering ---
    n_before = out.n_obs
    if min_counts is not None:
        sc.pp.filter_cells(out, min_counts=min_counts)
    if min_genes is not None:
        sc.pp.filter_cells(out, min_genes=min_genes)
    if max_counts is not None:
        out = out[out.obs["total_counts"] <= max_counts].copy()
    if max_genes is not None:
        out = out[out.obs["n_genes_by_counts"] <= max_genes].copy()
    print(f"Cell filtering: {n_before} → {out.n_obs} cells")

    n_genes_before = out.n_vars
    if min_cells is not None:
        sc.pp.filter_genes(out, min_cells=min_cells)
    print(f"Gene filtering: {n_genes_before} → {out.n_vars} genes")

    # --- 5. Mitochondrial filtering ---
    if max_pct_mito is not None and "pct_counts_MT" in out.obs.columns:
        n_before = out.n_obs
        out = out[out.obs["pct_counts_MT"] < max_pct_mito].copy()
        n_removed = n_before - out.n_obs
        if n_removed > 0:
            print(f"Mito filtering: removed {n_removed} cells with "
                  f">={max_pct_mito}% mitochondrial counts")

    # --- 6. QC Plots (pre-normalization) ---
    figures: dict[str, Any] = {}
    if plot:
        save_path_fn = None
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Histogram of counts and genes
        save_path_fn = str(save_dir / "qc_histograms.png") if save_dir else None
        fig_hist, _ = plot_qc_histograms(
            out,
            metrics=("total_counts", "n_genes_by_counts"),
            save_path=save_path_fn,
            dpi=dpi,
            show=show_plots,
        )
        figures["histograms"] = fig_hist

        # Violin plots
        violin_keys = ["n_genes_by_counts", "total_counts"]
        if "pct_counts_MT" in out.obs.columns and out.obs["pct_counts_MT"].sum() > 0:
            violin_keys.append("pct_counts_MT")
        save_path_fn = str(save_dir / "qc_violin.png") if save_dir else None
        fig_violin = plot_qc_violin(
            out,
            keys=tuple(violin_keys),
            groupby=batch_key,
            save_path=save_path_fn,
            dpi=dpi,
            show=show_plots,
        )
        figures["violin"] = fig_violin

        # Scatter: total_counts vs n_genes colored by mito %
        if "pct_counts_MT" in out.obs.columns:
            save_path_fn = str(save_dir / "qc_scatter.png") if save_dir else None
            fig_scatter, _ = plot_qc_scatter(
                out,
                save_path=save_path_fn,
                dpi=dpi,
                show=show_plots,
            )
            figures["scatter"] = fig_scatter

    # --- 7. Normalization ---
    # Ensure counts layer is up to date after filtering
    out.layers["counts"] = out.X.copy()

    if normalize:
        sc.pp.normalize_total(out, target_sum=target_sum)
        if log_transform:
            sc.pp.log1p(out)
            out.layers["log1p"] = out.X.copy()

    # --- 8. HVG selection ---
    if select_hvg:
        # Use counts layer for seurat_v3 flavor
        hvg_layer = "counts" if hvg_flavor == "seurat_v3" else None
        sc.pp.highly_variable_genes(
            out,
            n_top_genes=n_top_genes,
            flavor=hvg_flavor,
            layer=hvg_layer,
            subset=subset_hvg,
        )
        n_hvg = out.var["highly_variable"].sum() if not subset_hvg else out.n_vars
        print(f"HVG selection: {n_hvg} highly variable genes")

        # HVG plot
        if plot:
            save_path_fn = str(save_dir / "qc_hvg.png") if save_dir else None
            fig_hvg, _ = plot_hvg(
                out,
                save_path=save_path_fn,
                dpi=dpi,
                show=show_plots,
            )
            figures["hvg"] = fig_hvg

    # Store QC figures in uns for retrieval
    out.uns["qc_figures"] = figures

    print(f"\nQC workflow complete. Final shape: {out.shape[0]} cells × {out.shape[1]} genes")
    return out
