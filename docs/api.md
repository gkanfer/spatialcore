# API Overview

Core modules:

- `spatialcore.io`: AnnData, SpatialData, Xenium, Visium HD, and CODEX I/O helpers.
- `spatialcore.qc`: matrix selection, QC metrics, filters, and QC plots.
- `spatialcore.preprocessing`: normalization, HVG, PCA, neighbors, Leiden, UMAP, and spatial Leiden wrappers.
- `spatialcore.annotation`: KNN label transfer, CellTypist/scVI setup, and cell-type palettes.
- `spatialcore.alignment`: landmark alignment, affine transforms, simple image registration, coordinate/shape transforms, overlays, and alignment QC plots.
- `spatialcore.image`: SpatialData crop, image, labels, mask, and alignment helpers.
- `spatialcore.spatial`: nearest-neighbor distances, Squidpy spatial graphs, Moran's I, local Moran's I via Voyager, co-occurrence, proximity curves, distance models, and GraphCompass wrappers.
- `spatialcore.niches`: CellCharter grids, niche metrics, composition tables, and cosine similarity.
- `spatialcore.de`: pseudobulk, PyDESeq2, and Scanpy differential expression helpers.
- `spatialcore.enrichment`: GMT parsing, GSEA, AUCell, and leading-edge helpers.
- `spatialcore.plotting`: shared publication-style plotting functions.

Optional dependencies are imported lazily so basic imports stay lightweight.

## Spatial Backends

Squidpy is the preferred optional backend for spatial-neighbor graphs, co-occurrence curves, Ripley-style workflows, and global spatial autocorrelation. Use `spatial.compute_spatial_neighbors` before graph-dependent functions. Global Moran's I is exposed through `spatial.compute_spatial_autocorrelation(mode="moran")`; Geary's C is available through `mode="geary"`.

Local Moran's I is exposed through `spatial.compute_local_moran`, which delegates to `voyagerpy.spatial.local_moran` when Voyager is installed. If the installed Voyager package does not expose that function, the wrapper raises a clear TODO-style error rather than silently computing a different statistic.

Spatial Leiden remains in `preprocessing.run_spatial_leiden`. It expects a spatial-connectivity graph, usually `adata.obsp["spatial_connectivities"]` created by `spatial.compute_spatial_neighbors(..., key_added="spatial")`.

## Alignment Backends

`alignment.align_xenium_to_he` and `alignment.align_xenium_to_codex` are implemented as safe in-memory landmark-alignment helpers. They estimate transforms from paired landmarks and can optionally transform supplied Xenium coordinates. They do not load real Xenium, H&E, CODEX, AnnData, SpatialData, image, or table files, and they do not write outputs.

Supported alignment methods are manual landmark-based translation, rigid, similarity, and affine transforms through `alignment.compute_affine_transform`. `alignment.register_image_pair(method="translation")` provides optional translation-only image registration through scikit-image phase cross-correlation. Nonlinear/deformable registration is intentionally a placeholder that raises `NotImplementedError` until a safe backend and tests are added.

Use `alignment.apply_transform_to_coordinates` for coordinate arrays, `alignment.apply_transform_to_shapes` for shapely/geopandas objects, `alignment.overlay_images` for in-memory RGB overlays, and `alignment.plot_alignment_qc` for publication-style QC plots that save only when `save_path` is supplied.

## Design Choice

The package is intentionally function-based for now. AnnData and SpatialData already carry most workflow state, and small functions are easier to test with synthetic data. Project classes such as `SpatioCoreProject`, `XeniumProject`, or `VisiumHDProject` may be useful later for path management, manifests, and reproducible batch execution, but they are deferred until the workflow boundaries stabilize.
