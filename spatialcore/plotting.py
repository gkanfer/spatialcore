"""Publication-style plotting helpers used across the package."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import require_columns, save_figure_if_requested

PUBLICATION_RCPARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_publication_style(base_font_size: int = 8, dpi: int = 300) -> None:
    """Apply journal-friendly Matplotlib defaults globally."""
    params = dict(PUBLICATION_RCPARAMS)
    params.update(
        {
            "font.size": base_font_size,
            "axes.labelsize": base_font_size,
            "axes.titlesize": base_font_size + 1,
            "xtick.labelsize": max(base_font_size - 1, 6),
            "ytick.labelsize": max(base_font_size - 1, 6),
            "legend.fontsize": max(base_font_size - 1, 6),
            "savefig.dpi": dpi,
        }
    )
    mpl.rcParams.update(params)


@contextmanager
def publication_context(base_font_size: int = 8, dpi: int = 300):
    """Temporarily apply package publication style."""
    with mpl.rc_context(PUBLICATION_RCPARAMS):
        apply_publication_style(base_font_size=base_font_size, dpi=dpi)
        yield


def auto_figsize(n_panels: int, panel_size: tuple[float, float] = (3.0, 2.6), ncols: int = 2):
    """Compute a compact multi-panel figure size."""
    ncols = max(1, min(ncols, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    return ncols * panel_size[0], nrows * panel_size[1]


def finalize_figure(
    fig,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
):
    """Tighten, optionally save, optionally display, and return a figure."""
    fig.tight_layout()
    save_figure_if_requested(fig, save_path=save_path, dpi=dpi)
    if show:
        plt.show()
    return fig


def plot_spatial_scatter(
    df: pd.DataFrame,
    x: str = "x",
    y: str = "y",
    color: str | None = None,
    categorical: bool | None = None,
    ax=None,
    fig=None,
    size: float = 2.0,
    alpha: float = 0.8,
    palette: str | Sequence[str] = "tab20",
    cmap: str = "viridis",
    title: str | None = None,
    show_axes: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
):
    """Plot spatial coordinates with optional continuous or categorical color."""
    require_columns(df, [x, y], "df")
    if color is not None:
        require_columns(df, [color], "df")

    with publication_context(dpi=dpi):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
        elif fig is None:
            fig = ax.figure

        if color is None:
            ax.scatter(df[x], df[y], s=size, alpha=alpha, color="black", linewidths=0)
        else:
            if categorical is None:
                categorical = not pd.api.types.is_numeric_dtype(df[color])
            if categorical:
                cats = pd.Categorical(df[color])
                colors = sns.color_palette(palette, n_colors=len(cats.categories))
                lookup = dict(zip(cats.categories, colors))
                for cat in cats.categories:
                    sub = df[cats == cat]
                    ax.scatter(
                        sub[x],
                        sub[y],
                        s=size,
                        alpha=alpha,
                        color=lookup[cat],
                        linewidths=0,
                        label=str(cat),
                    )
                ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
            else:
                points = ax.scatter(
                    df[x],
                    df[y],
                    c=df[color],
                    s=size,
                    alpha=alpha,
                    cmap=cmap,
                    linewidths=0,
                )
                cbar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(color)

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title or "")
        if not show_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.set_xlabel(x)
            ax.set_ylabel(y)

        finalize_figure(fig, save_path=save_path, dpi=dpi, show=show)
        return fig, ax


def plot_volcano(
    df: pd.DataFrame,
    effect_col: str = "log2FoldChange",
    p_col: str = "padj",
    label_col: str | None = "gene",
    q_threshold: float = 0.05,
    effect_threshold: float = 1.0,
    ax=None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
):
    """Draw a compact differential-expression volcano plot."""
    require_columns(df, [effect_col, p_col], "df")
    plot_df = df.copy()
    pvals = pd.to_numeric(plot_df[p_col], errors="coerce").clip(lower=np.finfo(float).tiny)
    plot_df["_neg_log10_p"] = -np.log10(pvals)
    sig = (pvals < q_threshold) & (plot_df[effect_col].abs() >= effect_threshold)

    with publication_context(dpi=dpi):
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.4, 3.2), dpi=dpi)
        else:
            fig = ax.figure
        ax.scatter(plot_df[effect_col], plot_df["_neg_log10_p"], s=8, color="0.75", linewidths=0)
        ax.scatter(
            plot_df.loc[sig, effect_col],
            plot_df.loc[sig, "_neg_log10_p"],
            s=10,
            color="#c83e4d",
            linewidths=0,
        )
        ax.axvline(-effect_threshold, color="0.5", lw=0.7, ls="--")
        ax.axvline(effect_threshold, color="0.5", lw=0.7, ls="--")
        ax.axhline(-np.log10(q_threshold), color="0.5", lw=0.7, ls="--")
        ax.set_xlabel(effect_col)
        ax.set_ylabel(f"-log10({p_col})")
        if label_col and label_col in plot_df.columns:
            for _, row in plot_df.loc[sig].head(12).iterrows():
                ax.text(row[effect_col], row["_neg_log10_p"], str(row[label_col]), fontsize=6)
        finalize_figure(fig, save_path=save_path, dpi=dpi, show=show)
        return fig, ax


def plot_dotplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: str,
    color: str,
    ax=None,
    cmap: str = "viridis",
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
):
    """Generic publication-style dotplot for enrichment or marker summaries."""
    require_columns(df, [x, y, size, color], "df")
    with publication_context(dpi=dpi):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.5, max(2.5, 0.25 * df[y].nunique())), dpi=dpi)
        else:
            fig = ax.figure
        values = pd.to_numeric(df[size], errors="coerce").fillna(0)
        scaled = 20 + 180 * (values - values.min()) / (values.max() - values.min() + 1e-12)
        pts = ax.scatter(df[x], df[y], s=scaled, c=df[color], cmap=cmap, linewidths=0.2, edgecolors="0.2")
        fig.colorbar(pts, ax=ax, fraction=0.046, pad=0.04).set_label(color)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.tick_params(axis="x", rotation=45)
        finalize_figure(fig, save_path=save_path, dpi=dpi, show=show)
        return fig, ax


def plot_heatmap(
    data: pd.DataFrame,
    ax=None,
    cmap: str = "vlag",
    center: float | None = 0,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
    **kwargs,
):
    """Draw a compact heatmap and return figure and axis."""
    with publication_context(dpi=dpi):
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(3, 0.35 * data.shape[1]), max(2, 0.25 * data.shape[0])), dpi=dpi)
        else:
            fig = ax.figure
        sns.heatmap(data, cmap=cmap, center=center, ax=ax, **kwargs)
        finalize_figure(fig, save_path=save_path, dpi=dpi, show=show)
        return fig, ax
