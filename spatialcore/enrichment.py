"""GSEA and pathway-enrichment helpers."""

from __future__ import annotations

from itertools import chain, repeat
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .utils import optional_import, require_columns


def clean_term_label(term: str) -> str:
    """Clean common database prefixes and separators from enrichment terms."""
    label = str(term)
    for prefix in ("HALLMARK_", "REACTOME_", "GOBP_", "GO_", "WP_"):
        label = label.replace(prefix, "")
    return label.replace("_", " ").title()


def gmt_to_decoupler(file: str | Path) -> pd.DataFrame:
    """Read a GMT file into decoupler's `source`, `target` format."""
    records = []
    with Path(file).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term, _, *genes = parts
            records.extend(zip(repeat(term), genes))
    return pd.DataFrame(records, columns=["source", "target"])


def run_gsea_prerank(
    ranking: pd.DataFrame | pd.Series,
    gene_sets: str | Path | dict[str, list[str]],
    gene_col: str = "gene",
    score_col: str = "score",
    min_size: int = 5,
    max_size: int = 500,
    permutation_num: int = 1000,
    seed: int = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run gseapy prerank and return a result table."""
    gp = optional_import("gseapy", "modeling")
    if isinstance(ranking, pd.Series):
        rnk = ranking.sort_values(ascending=False)
    else:
        require_columns(ranking, [gene_col, score_col], "ranking")
        rnk = ranking[[gene_col, score_col]].dropna().sort_values(score_col, ascending=False)
    result = gp.prerank(
        rnk=rnk,
        gene_sets=gene_sets,
        min_size=min_size,
        max_size=max_size,
        permutation_num=permutation_num,
        seed=seed,
        outdir=None,
        **kwargs,
    )
    return result.res2d.copy()


def score_pathways_aucell(
    adata: Any,
    gene_set_table: pd.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    layer: str | None = None,
    obsm_key: str = "pathway_scores",
):
    """Score pathways with decoupler AUCell and store results in `adata.obsm`."""
    dc = optional_import("decoupler", "modeling")
    require_columns(gene_set_table, [source_col, target_col], "gene_set_table")
    out = adata.copy()
    mat = out.layers[layer] if layer is not None else out.X
    scores = dc.run_aucell(mat=mat, net=gene_set_table, source=source_col, target=target_col)
    out.obsm[obsm_key] = scores[0] if isinstance(scores, tuple) else scores
    return out


def leading_edge_table(
    gsea_res: pd.DataFrame,
    term_col: str = "Term",
    leading_edge_col: str = "Lead_genes",
    separator: str = ";",
) -> pd.DataFrame:
    """Expand a GSEA leading-edge gene column into a tidy table."""
    require_columns(gsea_res, [term_col, leading_edge_col], "gsea_res")
    records = []
    for _, row in gsea_res.iterrows():
        genes = str(row[leading_edge_col]).replace(",", separator).split(separator)
        records.extend({"term": row[term_col], "gene": gene.strip()} for gene in genes if gene.strip())
    return pd.DataFrame.from_records(records)


def remove_pathways(df: pd.DataFrame, pathways: Sequence[str], term_col: str = "source") -> pd.DataFrame:
    """Remove selected pathway rows from a long pathway table."""
    if term_col not in df.columns:
        raise KeyError(f"`{term_col}` was not found.")
    return df.loc[~df[term_col].isin(pathways)].copy()


def compute_pathway_score(*args: Any, **kwargs: Any):
    """Alias-friendly corrected spelling for pathway scoring."""
    return score_pathways_aucell(*args, **kwargs)


def campute_pathway_score(*args: Any, **kwargs: Any):
    """Backward-compatible alias for notebook spelling."""
    return compute_pathway_score(*args, **kwargs)
