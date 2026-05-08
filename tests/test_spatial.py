import pandas as pd

from spatialcore.spatial import compute_nearest_neighbor_distances, run_celltype_distance_analysis


def test_compute_nearest_neighbor_distances(synthetic_adata):
    out = compute_nearest_neighbor_distances(synthetic_adata)
    assert "nearest_neighbor_distance" in out.obs
    assert out.obs["nearest_neighbor_distance"].min() >= 0


def test_run_celltype_distance_analysis():
    df = pd.DataFrame(
        {
            "cell_type": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "condition": ["ctrl", "ctrl", "tx", "tx"] * 2,
            "timepoint": [4, 4, 4, 4] * 2,
            "batch_pair": ["b1", "b2", "b1", "b2"] * 2,
            "distance_to_target": [10, 12, 20, 22, 5, 6, 9, 10],
        }
    )
    results, diagnostics = run_celltype_distance_analysis(df, distance_threshold=100)
    assert not results.empty
    assert "q_value" in results
    assert "model_fits" in diagnostics
