"""Report and output-table helpers that save only when explicitly requested."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .utils import save_table_if_requested


def sample_summary(
    adata: Any,
    sample_col: str,
    counts_col: str = "total_counts",
    genes_col: str = "n_genes_by_counts",
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Create a sample-level QC summary table."""
    for col in (sample_col, counts_col, genes_col):
        if col not in adata.obs:
            raise KeyError(f"`adata.obs['{col}']` was not found.")
    df = (
        adata.obs.groupby(sample_col, observed=True)
        .agg(
            n_obs=(counts_col, "size"),
            mean_counts=(counts_col, "mean"),
            median_counts=(counts_col, "median"),
            mean_genes=(genes_col, "mean"),
            median_genes=(genes_col, "median"),
        )
        .reset_index()
    )
    save_table_if_requested(df, save_path)
    return df


def table_with_metadata(
    table: pd.DataFrame,
    metadata: Mapping[str, Any],
    save_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Return a table plus metadata DataFrame, and optionally save the main table."""
    metadata_df = pd.DataFrame({"parameter": list(metadata.keys()), "value": list(metadata.values())})
    save_table_if_requested(table, save_path)
    return {"table": table, "metadata": metadata_df}
