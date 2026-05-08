"""Template for Visium HD preprocessing.

This file is not run by package setup or tests.
"""

from spatialcore.io import read_visium_hd
from spatialcore.preprocessing import normalize_log1p, run_leiden_umap, run_pca_neighbors, select_hvgs


def visiumhd_preprocessing_template(visium_path):
    adata = read_visium_hd(visium_path)
    adata = normalize_log1p(adata)
    adata = select_hvgs(adata, subset=True)
    adata = run_pca_neighbors(adata)
    adata = run_leiden_umap(adata)
    return adata
