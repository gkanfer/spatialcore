"""Template for GSEA analysis."""

from spatialcore.enrichment import leading_edge_table, run_gsea_prerank


def gsea_template(ranking_df, gmt_path):
    gsea_results = run_gsea_prerank(ranking_df, gene_sets=gmt_path)
    leading_edge = leading_edge_table(gsea_results)
    return {"gsea": gsea_results, "leading_edge": leading_edge}
