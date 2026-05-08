"""Cell composition, log-odds, and summary table helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from .utils import require_columns


def format_pvalue(pvalue: float) -> str:
    """Format p-values as star labels."""
    if pd.isna(pvalue):
        return ""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def composition_table(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    category_col: str,
    count_col: str | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Create a grouped composition count/proportion table."""
    require_columns(df, [*group_cols, category_col], "df")
    if count_col is not None:
        require_columns(df, [count_col], "df")
        counts = df.groupby([*group_cols, category_col], observed=True)[count_col].sum().reset_index(name="count")
    else:
        counts = df.groupby([*group_cols, category_col], observed=True).size().reset_index(name="count")
    if normalize:
        total = counts.groupby(list(group_cols), observed=True)["count"].transform("sum")
        counts["proportion"] = counts["count"] / total.replace(0, np.nan)
    return counts


def add_logit_proportion(
    df: pd.DataFrame,
    proportion_col: str = "proportion",
    output_col: str = "logit_proportion",
    eps: float = 1e-4,
) -> pd.DataFrame:
    """Add clipped logit-transformed proportions."""
    require_columns(df, [proportion_col], "df")
    out = df.copy()
    prop = out[proportion_col].clip(eps, 1 - eps)
    out[output_col] = np.log(prop / (1 - prop))
    return out


def pairwise_composition_tests(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    comparisons: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    """Run pairwise Welch t-tests for composition summaries."""
    require_columns(df, [value_col, group_col], "df")
    records = []
    for group, reference in comparisons:
        vals = df.loc[df[group_col].astype(str) == str(group), value_col].dropna()
        ref = df.loc[df[group_col].astype(str) == str(reference), value_col].dropna()
        if vals.empty or ref.empty:
            continue
        stat = ttest_ind(vals, ref, equal_var=False, nan_policy="omit")
        records.append(
            {
                "group": group,
                "reference": reference,
                "mean_group": vals.mean(),
                "mean_reference": ref.mean(),
                "mean_difference": vals.mean() - ref.mean(),
                "pvalue": float(stat.pvalue),
            }
        )
    out = pd.DataFrame.from_records(records)
    if not out.empty:
        _, qvals, _, _ = multipletests(out["pvalue"], method="fdr_bh")
        out["qvalue"] = qvals
        out["significance"] = out["qvalue"].apply(format_pvalue)
    return out


def celltype_stacked_table(
    adata,
    sample_col: str,
    celltype_col: str = "cell_type",
) -> pd.DataFrame:
    """Create a sample by cell-type proportion table from AnnData obs."""
    return composition_table(adata.obs.reset_index(drop=True), [sample_col], celltype_col, normalize=True)
