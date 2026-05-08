# Dependency Strategy

One unified environment is not expected to be reliable for all workflows.

Recommended environments:

- `environment.yml`: recommended default for most users; core AnnData, Scanpy, plotting, statistics, and notebooks.
- `environment-dev.yml`: package development, linting, and synthetic tests.
- `environment-xenium.yml`: Xenium, SpatialData, Squidpy, CellTypist, scVI, and image/table helpers.
- `environment-visiumhd.yml`: Visium HD, SpatialData, Squidpy, bin2cell, voyagerpy, scVI, local Moran's I, and spatial Leiden.
- `environment-codex-he.yml`: CODEX/H&E alignment, SpatialData image/shape helpers, scikit-image, tifffile, imagecodecs, geopandas, and shapely.
- `environment-modeling.yml`: PyDESeq2, GSEA, decoupler, sccoda, TensorFlow, numpyro, and PyMC.
- `environment-all.yml`: experimental full-stack template only; useful as a dependency inventory, but not recommended as the default environment.

Use `environment.yml` first unless a workflow needs optional heavy dependencies.
Use `environment-xenium.yml` for Xenium-specific SpatialData, Squidpy,
CellTypist, and scVI workflows. Use `environment-visiumhd.yml` for Visium HD,
bin2cell, Voyager/local Moran's I, and spatial Leiden workflows. Use
`environment-codex-he.yml` for in-memory alignment helpers, SpatialData image
workflows, CODEX/H&E masks, and overlay QC.

Squidpy is the preferred optional backend for spatial-neighbor graphs, co-occurrence, Ripley-style statistics, and global Moran's I. The core package still imports without Squidpy because these workflows are lazy-loaded through `spatialcore.spatial`.

Local Moran's I is routed through Voyager when `voyagerpy.spatial.local_moran` is available. This keeps the implementation close to the notebook workflow while avoiding a hard dependency in the core environment.

Main conflict risks:

- TensorFlow/sccoda pinning around TensorFlow 2.12.
- SpatialData stacks commonly using Python 3.12 and NumPy 2.x.
- RAPIDS/CuPy/RMM requiring CUDA-compatible systems.
- Napari GUI packages adding Qt constraints.
- scVI/PyTorch and older LR packages depending on different Python/NumPy ranges.
- GraphCompass availability and build requirements may vary by platform.

Package code uses lazy imports and optional dependency groups to keep core imports stable.

Optional dependency groups in `pyproject.toml` are imported only when their
functions are called:

- `spatial`: SpatialData, Squidpy, geopandas/shapely, scikit-image, and tifffile.
- `codex-he`: CODEX/H&E alignment and image/shape dependencies.
- `xenium`: Xenium-oriented SpatialData, CellTypist, scVI, Squidpy, and stLearn.
- `visiumhd`: Visium HD, bin2cell, Voyager, Squidpy, and spatialleiden.
- `modeling`: PyDESeq2, GSEA/decoupler, sccoda, TensorFlow, numpyro, and PyMC.
- `niche`: CellCharter, GraphCompass, TACCO, and scib.

A single all-in-one environment is not expected to be realistic for routine use.
It is likely to hit conflicts among TensorFlow/sccoda, PyTorch/scVI, SpatialData
and Squidpy, GPU libraries, and platform-specific packages. Prefer modular
workflow environments and keep the all-in-one file as an experimental template.
