from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialcore import spatial


def test_compute_spatial_neighbors_uses_squidpy_conventions(monkeypatch, synthetic_adata):
    calls = {}

    def fake_spatial_neighbors(adata, **kwargs):
        calls.update(kwargs)
        key = kwargs["key_added"]
        adata.obsp[f"{key}_connectivities"] = sparse.eye(adata.n_obs, format="csr")
        adata.obsp[f"{key}_distances"] = sparse.eye(adata.n_obs, format="csr")
        adata.uns[key] = {"params": kwargs}

    fake_sq = SimpleNamespace(gr=SimpleNamespace(spatial_neighbors=fake_spatial_neighbors))
    monkeypatch.setattr(spatial, "optional_import", lambda *_args, **_kwargs: fake_sq)

    out = spatial.compute_spatial_neighbors(
        synthetic_adata,
        n_neighs=3,
        key_added="mock_spatial",
        inplace=False,
    )

    assert calls["spatial_key"] == "spatial"
    assert calls["n_neighs"] == 3
    assert "mock_spatial_connectivities" in out.obsp
    assert "mock_spatial_distances" in out.obsp
    assert "mock_spatial_connectivities" not in synthetic_adata.obsp


def test_compute_spatial_autocorrelation_returns_table(monkeypatch, synthetic_adata):
    expected = pd.DataFrame({"I": [0.2], "pval_norm": [0.05]}, index=["GeneA"])
    synthetic_adata.obsp["spatial_connectivities"] = sparse.eye(synthetic_adata.n_obs, format="csr")
    calls = {}

    def fake_spatial_autocorr(adata, **kwargs):
        calls.update(kwargs)
        return expected

    fake_sq = SimpleNamespace(gr=SimpleNamespace(spatial_autocorr=fake_spatial_autocorr))
    monkeypatch.setattr(spatial, "optional_import", lambda *_args, **_kwargs: fake_sq)

    result = spatial.compute_spatial_autocorrelation(
        synthetic_adata,
        genes=["GeneA"],
        mode="moran",
        connectivity_key="spatial_connectivities",
    )

    assert calls["mode"] == "moran"
    assert calls["genes"] == ["GeneA"]
    assert calls["connectivity_key"] == "spatial_connectivities"
    pd.testing.assert_frame_equal(result, expected)


def test_compute_spatial_autocorrelation_requires_graph(synthetic_adata):
    with pytest.raises(KeyError, match="compute_spatial_neighbors"):
        spatial.compute_spatial_autocorrelation(synthetic_adata, genes=["GeneA"])


def test_compute_co_occurrence_returns_tidy_table(monkeypatch, synthetic_adata):
    occ = np.array(
        [
            [[1.0, 1.2], [0.8, 0.9]],
            [[0.7, 0.6], [1.1, 1.4]],
        ]
    )
    distances = np.array([0.0, 10.0, 20.0])

    def fake_co_occurrence(*_args, **_kwargs):
        return occ, distances

    fake_sq = SimpleNamespace(gr=SimpleNamespace(co_occurrence=fake_co_occurrence))
    monkeypatch.setattr(spatial, "optional_import", lambda *_args, **_kwargs: fake_sq)

    result = spatial.compute_co_occurrence(
        synthetic_adata,
        cluster_key="cell_type",
        source="A",
        interval=np.array([0, 10, 20]),
    )

    assert list(result.columns) == ["source", "partner", "distance", "ratio"]
    assert set(result["source"]) == {"A"}
    assert set(result["partner"]) == {"A", "B"}
    assert result["distance"].tolist() == [10.0, 20.0, 10.0, 20.0]


def test_compute_local_moran_uses_voyager_backend(monkeypatch, synthetic_adata):
    calls = []

    def fake_local_moran(adata, gene, graph_name, **_kwargs):
        calls.append((gene, graph_name))
        adata.obsm["local_moran"] = pd.DataFrame(
            {gene: np.arange(adata.n_obs)},
            index=adata.obs_names,
        )

    fake_vp = SimpleNamespace(spatial=SimpleNamespace(local_moran=fake_local_moran))
    monkeypatch.setattr(spatial, "optional_import", lambda *_args, **_kwargs: fake_vp)

    out = spatial.compute_local_moran(synthetic_adata, genes="GeneA", graph_name="spatial")

    assert calls == [("GeneA", "spatial")]
    assert "local_moran" in out.obsm
    assert "local_moran" not in synthetic_adata.obsm


def test_compute_local_moran_has_clear_todo_when_backend_lacks_function(
    monkeypatch,
    synthetic_adata,
):
    fake_vp = SimpleNamespace(spatial=SimpleNamespace())
    monkeypatch.setattr(spatial, "optional_import", lambda *_args, **_kwargs: fake_vp)

    with pytest.raises(NotImplementedError, match="Local Moran's I"):
        spatial.compute_local_moran(synthetic_adata, genes="GeneA")
