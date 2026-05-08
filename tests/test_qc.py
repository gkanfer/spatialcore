from spatialcore.qc import compute_qc_metrics, flag_mito_genes


def test_compute_qc_metrics_synthetic(synthetic_adata):
    qc_df, summary_df, metadata = compute_qc_metrics(synthetic_adata)
    assert qc_df.shape[0] == synthetic_adata.n_obs
    assert "n_reads" in qc_df
    assert metadata["n_cells"] == synthetic_adata.n_obs
    assert not summary_df.empty


def test_flag_mito_genes_synthetic(synthetic_adata):
    out = flag_mito_genes(synthetic_adata, inplace=False)
    assert "MT" in out.var
    assert "pct_counts_MT" in out.obs
