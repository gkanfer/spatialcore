from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


@pytest.fixture
def synthetic_adata():
    ad = pytest.importorskip("anndata")
    X = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 3, 0],
                [0, 2, 1, 0],
                [5, 0, 0, 2],
                [0, 1, 0, 4],
                [2, 2, 2, 2],
                [0, 0, 1, 1],
            ],
            dtype=float,
        )
    )
    obs = pd.DataFrame(
        {
            "sample": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "cell_type": ["A", "B", "A", "B", "A", "B"],
            "condition": ["ctrl", "ctrl", "tx", "tx", "ctrl", "tx"],
            "timepoint": [4, 4, 4, 4, 8, 8],
            "batch_pair": ["b1", "b1", "b2", "b2", "b3", "b3"],
        },
        index=[f"cell{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["mt-Nd1", "GeneA", "GeneB", "GeneC"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    adata.obsm["spatial"] = np.array([[0, 0], [1, 0], [5, 0], [6, 0], [0, 5], [1, 5]], dtype=float)
    return adata
