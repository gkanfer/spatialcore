# spatialcore

`spatialcore` is an early-stage open-source Python package for reusable spatial omics workflows derived from the notebooks and helper scripts in this repository.

The first implementation focuses on reusable function APIs for:

- AnnData and SpatialData I/O
- Xenium and Visium HD analysis scaffolds
- CODEX and H&E SpatialData/alignment helpers
- quality control and preprocessing
- cell annotation and label transfer
- spatial distances, Squidpy-backed spatial graphs, Moran's I, proximity curves, neighborhoods, and niches
- differential expression, GSEA, and pathway scoring
- publication-style plotting and report-table helpers

The original notebooks and example scripts are treated as read-only references. Package functions are designed to work with explicit user-provided objects and paths; plotting and table functions do not save files unless a `save_path` or explicit output path is provided.

## Install

Core package:

```bash
pip install -e .
```

Optional workflow groups:

```bash
pip install -e ".[spatial]"
pip install -e ".[codex-he]"
pip install -e ".[xenium]"
pip install -e ".[visiumhd]"
pip install -e ".[modeling]"
pip install -e ".[niche]"
```

The full project stack is not expected to fit reliably into one environment. See `docs/dependencies.md` and the environment templates for a modular strategy.

## Example Notebook

`examples/spatialcore_function_testing_notebook.ipynb` is a synthetic-data walkthrough for the main package APIs. It demonstrates QC, preprocessing, plotting, Squidpy spatial graph wrappers, Moran's I/local Moran's I entry points, spatial Leiden, niches, DGE/GSEA-style tables, LR summaries, and CODEX/H&E-style mock landmark alignment and image handling without loading real project data or saving outputs by default.

## Alignment Status

Xenium-to-H&E and Xenium-to-CODEX alignment are implemented as safe, in-memory,
landmark-based helpers in `spatialcore.alignment`. Supported transforms are
translation, rigid, similarity, and affine. Overlay QC and coordinate/shape
transform helpers are included. Full nonlinear/deformable registration is not
implemented yet; those code paths raise explicit TODO errors.

## Design Principles

- Notebook-derived workflows are converted into small, testable functions.
- Heavy workflow dependencies are imported lazily.
- Alignment functions operate on in-memory arrays, landmarks, coordinates, and shapes; they do not read or write real data.
- Squidpy is used as the preferred optional backend for spatial graphs, co-occurrence, Ripley-style workflows, and global Moran's I.
- Scientific plots default to journal-style sizing, fonts, legends, and 300 DPI.
- Plotting functions display by default, return figure/axes objects, and save only when a path is supplied.
- Tests use synthetic or mock data only.
- The public API is function-based for now; project classes can be added later once workflow boundaries stabilize.

## Read-Only References

These files and directories are source references and should not be modified by package implementation work:

- `spatiocore_reference_notebook.ipynb`
- `notebook_list.txt`
- `mamba_install.txt`
- `agent_instraction.txt`
- `example_notebooks_from_github/`
