"""I/O helpers for AnnData, SpatialData, Xenium, Visium HD, and CODEX workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .utils import optional_import


def read_h5ad(path: str | Path, backed: str | None = None):
    """Read an AnnData object from H5AD."""
    sc = optional_import("scanpy", "core")
    return sc.read_h5ad(path, backed=backed)


def write_h5ad(adata: Any, path: str | Path, **kwargs: Any) -> None:
    """Write AnnData to H5AD when explicitly called."""
    adata.write_h5ad(str(path), **kwargs)


def read_spatialdata(path: str | Path):
    """Read a SpatialData Zarr store."""
    sd = optional_import("spatialdata", "spatial")
    return sd.read_zarr(str(path))


def write_spatialdata(sdata: Any, path: str | Path, overwrite: bool = False, **kwargs: Any) -> None:
    """Write a SpatialData object when explicitly called."""
    if hasattr(sdata, "write"):
        sdata.write(str(path), overwrite=overwrite, **kwargs)
    else:
        raise TypeError("Expected a SpatialData-like object with a `.write` method.")


def convert_visium_tissue_positions(
    visium_path: str | Path,
    output_csv: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert Visium `tissue_positions.parquet` to Scanpy-compatible CSV."""
    visium_path = Path(visium_path)
    parquet_path = visium_path / "spatial" / "tissue_positions.parquet"
    csv_path = Path(output_csv) if output_csv is not None else visium_path / "spatial" / "tissue_positions_list.csv"
    if csv_path.exists() and not overwrite:
        return csv_path
    if not parquet_path.exists():
        raise FileNotFoundError(f"Could not find {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df.to_csv(csv_path, index=False)
    return csv_path


def read_visium_hd(
    path: str | Path,
    convert_positions: bool = False,
    source_image_path: str | Path | None = None,
    **kwargs: Any,
):
    """Read a Visium/Visium HD folder with Scanpy.

    Set `convert_positions=True` to explicitly create Scanpy's legacy
    `spatial/tissue_positions_list.csv` from `tissue_positions.parquet`.
    """
    sc = optional_import("scanpy", "core")
    if convert_positions:
        convert_visium_tissue_positions(path)
    return sc.read_visium(path=str(path), source_image_path=source_image_path, **kwargs)


def read_xenium_spatialdata(path: str | Path, **kwargs: Any):
    """Read Xenium output as SpatialData via `spatialdata_io.xenium`."""
    spatialdata_io = optional_import("spatialdata_io", "spatial")
    return spatialdata_io.xenium(str(path), **kwargs)


def one_row_per_cell(adata: Any, cell_id_col: str = "cell_id"):
    """Keep one observation row per physical cell when a cell id column is present."""
    if cell_id_col not in adata.obs:
        return adata
    keep = ~adata.obs[cell_id_col].astype(str).duplicated()
    return adata[keep].copy()


def concat_anndata_by_sample(
    paths: Sequence[str | Path],
    sample_metadata: Mapping[str, Mapping[str, Any]] | pd.DataFrame | None = None,
    sample_key: str = "sample",
    cell_id_col: str = "cell_id",
    index_unique: str | None = "-",
):
    """Read, annotate, one-row-clean, and concatenate multiple H5AD files."""
    anndata = optional_import("anndata", "core")
    adatas = []
    metadata_df = None
    if isinstance(sample_metadata, pd.DataFrame):
        metadata_df = sample_metadata.copy()
        if sample_key not in metadata_df.columns:
            metadata_df[sample_key] = metadata_df.index.astype(str)

    for path in paths:
        ad = read_h5ad(path)
        sample = Path(path).stem.split("__")[0]
        ad.obs[sample_key] = sample
        if sample_metadata is not None:
            if metadata_df is not None:
                rows = metadata_df[metadata_df[sample_key].astype(str) == sample]
                if not rows.empty:
                    for col, value in rows.iloc[0].items():
                        ad.obs[col] = value
            elif sample in sample_metadata:
                for col, value in sample_metadata[sample].items():
                    ad.obs[col] = value
        adatas.append(one_row_per_cell(ad, cell_id_col=cell_id_col))

    return anndata.concat(adatas, label=sample_key, keys=[a.obs[sample_key].iloc[0] for a in adatas], index_unique=index_unique)


def build_codex_spatialdata(
    image_array: Any,
    table: pd.DataFrame,
    marker_cols: Sequence[str],
    x_col: str = "x",
    y_col: str = "y",
    radius: float = 2.0,
    image_name: str = "codex",
    shapes_name: str = "cells",
    table_name: str = "table",
):
    """Build a basic CODEX SpatialData object from an image array and cell table."""
    anndata = optional_import("anndata", "core")
    gpd = optional_import("geopandas", "spatial")
    sd = optional_import("spatialdata", "spatial")
    models = optional_import("spatialdata.models", "spatial")
    shapely_geometry = optional_import("shapely.geometry", "spatial")

    missing = [col for col in [x_col, y_col, *marker_cols] if col not in table.columns]
    if missing:
        raise KeyError(f"CODEX table is missing required columns: {missing}")

    image_element = models.Image2DModel.parse(image_array)
    geometry = [shapely_geometry.Point(xy).buffer(radius) for xy in table[[x_col, y_col]].to_numpy()]
    gdf = gpd.GeoDataFrame(table.drop(columns=list(marker_cols), errors="ignore").copy(), geometry=geometry)
    shapes = models.ShapesModel.parse(gdf)

    X = table.loc[:, marker_cols].to_numpy()
    obs = table.drop(columns=list(marker_cols), errors="ignore").copy()
    adata = anndata.AnnData(X=X, obs=obs)
    adata.var_names = list(marker_cols)
    adata_table = models.TableModel.parse(adata, region=shapes_name, region_key="region", instance_key=None)
    return sd.SpatialData(images={image_name: image_element}, shapes={shapes_name: shapes}, tables={table_name: adata_table})
