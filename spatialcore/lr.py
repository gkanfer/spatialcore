"""Ligand-receptor result standardization and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .plotting import plot_spatial_scatter
from .utils import require_columns, save_table_if_requested


def standardize_lr_table(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize notebook LR result tables to `lr_pair` and `lr_score_adjust`."""
    out = df.copy()
    pair_candidates = ["lr_pair", "ligand_receptor", "ligand_receptor_pair", "ligand_receptor_pairs", "interacting_pair"]
    pair_col = next((c for c in pair_candidates if c in out.columns), None)
    if pair_col is None and {"ligand", "receptor"}.issubset(out.columns):
        out["lr_pair"] = out["ligand"].astype(str) + "_" + out["receptor"].astype(str)
    elif pair_col and pair_col != "lr_pair":
        out = out.rename(columns={pair_col: "lr_pair"})

    if "lr_score_adjust" not in out.columns:
        if {"lr_score", "n_spots"}.issubset(out.columns):
            out["lr_score_adjust"] = pd.to_numeric(out["lr_score"], errors="coerce") * np.log1p(
                pd.to_numeric(out["n_spots"], errors="coerce")
            )
        elif {"n_spots_sig", "n_spots"}.issubset(out.columns):
            n_spots = pd.to_numeric(out["n_spots"], errors="coerce").replace(0, np.nan)
            out["lr_score"] = pd.to_numeric(out["n_spots_sig"], errors="coerce") / n_spots
            out["lr_score_adjust"] = out["lr_score"] * np.log1p(n_spots)
    require_columns(out, ["lr_pair", "lr_score_adjust"], "LR table")
    return out


def rank_lr_pairs(
    df: pd.DataFrame,
    condition_col: str | None = None,
    top_n: int = 50,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Rank ligand-receptor pairs globally or by condition."""
    out = standardize_lr_table(df)
    if condition_col is None:
        ranked = (
            out.groupby("lr_pair", as_index=False)["lr_score_adjust"]
            .mean()
            .rename(columns={"lr_score_adjust": "mean_lr_score_adjust"})
            .sort_values("mean_lr_score_adjust", ascending=False)
            .head(top_n)
        )
        ranked["rank"] = np.arange(1, len(ranked) + 1)
    else:
        require_columns(out, [condition_col], "LR table")
        ranked = (
            out.groupby([condition_col, "lr_pair"], as_index=False)["lr_score_adjust"]
            .mean()
            .rename(columns={"lr_score_adjust": "mean_lr_score_adjust"})
            .sort_values([condition_col, "mean_lr_score_adjust"], ascending=[True, False])
        )
        ranked["rank_within_condition"] = ranked.groupby(condition_col).cumcount() + 1
        ranked = ranked[ranked["rank_within_condition"] <= top_n].copy()
    save_table_if_requested(ranked, save_path)
    return ranked


def plot_spatial_lr_analysis(
    df: pd.DataFrame,
    x: str = "x",
    y: str = "y",
    score_col: str = "lr_scores",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Plot LR scores in spatial coordinates."""
    return plot_spatial_scatter(
        df,
        x=x,
        y=y,
        color=score_col,
        categorical=False,
        cmap="Greens",
        size=3,
        save_path=save_path,
        show=show,
    )


def obs_sig_lr(adata, lr_top: Sequence[str], obs_prefix: str = "lr_sig") -> pd.DataFrame:
    """Summarize LR significance columns from AnnData obs for selected LR pairs."""
    cols = [col for col in adata.obs.columns if any(pair in col for pair in lr_top)]
    return adata.obs[cols].add_prefix(f"{obs_prefix}_").copy()
