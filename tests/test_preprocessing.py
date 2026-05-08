from spatialcore.preprocessing import voyager_transform


def test_voyager_transform_adds_thresholds(synthetic_adata):
    adata = synthetic_adata.copy()
    adata.obs["total_counts"] = [1, 2, 3, 4, 5, 6]
    adata.obs["n_genes_by_counts"] = [1, 1, 2, 2, 3, 3]
    out = voyager_transform(adata)
    assert "voyager_keep" in out.obs
    assert "voyager_thresholds" in out.uns
