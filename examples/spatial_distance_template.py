"""Template for spatial distance analysis."""

from spatialcore.spatial import (
    celltype_distance_table,
    compute_nearest_neighbor_distances,
    run_celltype_distance_analysis,
)


def spatial_distance_template(adata, target_mask):
    adata = compute_nearest_neighbor_distances(adata)
    distance_df = celltype_distance_table(adata, target_mask=target_mask, cell_type_key="cell_type")
    results, diagnostics = None, None
    required = {"cell_type", "condition", "timepoint", "batch_pair", "distance_to_target"}
    if required.issubset(distance_df.columns):
        results, diagnostics = run_celltype_distance_analysis(distance_df)
    return {"adata": adata, "distances": distance_df, "results": results, "diagnostics": diagnostics}
