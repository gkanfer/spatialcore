from spatialcore.io import one_row_per_cell


def test_one_row_per_cell(synthetic_adata):
    adata = synthetic_adata.copy()
    adata.obs["cell_id"] = ["a", "a", "b", "c", "c", "d"]
    out = one_row_per_cell(adata)
    assert out.n_obs == 4
