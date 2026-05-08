# Workflow Notes

## QC and Preprocessing

Use `qc.compute_qc_metrics` on in-memory AnnData objects, then plot with `plot_qc_histograms`, `plot_qc_boxplots`, or `plot_qc_spatial`. Use `preprocessing.normalize_log1p`, `select_hvgs`, `run_pca_neighbors`, and `run_leiden_umap` for standard Scanpy-style workflows.

## SpatialData Display

Use `image.crop_spatialdata` for bounding-box crops and `annotation.rebuild_cropped_table` to repair table metadata after cropping. Use `image.plot_spatial_crop` for image/label display without saving unless `save_path` is provided.

## Xenium, CODEX, And H&E Alignment

Use `alignment.align_xenium_to_he` and `alignment.align_xenium_to_codex` for manual landmark-based alignment. These functions estimate a transform from paired landmarks and optionally apply it to in-memory Xenium coordinates. They do not read or write real data.

Use `alignment.compute_affine_transform` for translation, rigid, similarity, or affine landmark transforms. Use `alignment.apply_transform_to_coordinates` and `alignment.apply_transform_to_shapes` to apply affine transforms to coordinates and shape-like objects. Use `alignment.overlay_images` and `alignment.plot_alignment_qc` for overlay QC, with `save_path=None` by default.

Image-to-image registration is currently limited to optional translation estimation through scikit-image phase cross-correlation. Nonlinear/deformable registration is not implemented yet and raises `NotImplementedError`.

## Spatial Distances

Use `spatial.compute_nearest_neighbor_distances` and `spatial.celltype_distance_table` for distance summaries. Use `spatial.run_celltype_distance_analysis` for cell-type stratified distance comparisons.

## Squidpy Spatial Graphs And Autocorrelation

Use `spatial.compute_spatial_neighbors` to build Squidpy spatial graphs for Xenium, Visium HD, and general spatial transcriptomics AnnData objects. The default `key_added="spatial"` stores `spatial_connectivities` and `spatial_distances`, matching Squidpy conventions and the spatial Leiden workflow.

Use `spatial.compute_spatial_autocorrelation(..., mode="moran")` for global Moran's I and `mode="geary"` for Geary's C. Use `spatial.compute_local_moran` for Voyager-backed local Moran's I when `voyagerpy.spatial.local_moran` is available. Use `spatial.compute_co_occurrence` for Squidpy co-occurrence/proximity curves similar to the highlighted spatial-proximity notebooks.

`preprocessing.run_spatial_leiden` expects an existing spatial-connectivity matrix. A typical workflow is:

```python
adata = spatial.compute_spatial_neighbors(adata, coord_type="generic", radius=50)
adata = preprocessing.run_spatial_leiden(adata, key_added="spatial_leiden")
```

## Niches

Use `niches.run_cellcharter_grid` when the `cellcharter` optional environment is active. Use `niches.niche_composition_table` and `niches.evaluate_niche_result` for interpretation.

## DE and Enrichment

Use `de.make_pseudobulk_replicates`, `de.run_pydeseq2`, `enrichment.run_gsea_prerank`, and `enrichment.score_pathways_aucell` with explicit in-memory objects and paths.

## Example Notebook

`examples/spatialcore_function_testing_notebook.ipynb` is a user-facing synthetic-data walkthrough. It demonstrates the main package APIs without loading real Xenium, Visium HD, CODEX, H&E, AnnData, SpatialData, or image files, and it does not save plots or tables by default.
