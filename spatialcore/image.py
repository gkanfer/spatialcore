"""SpatialData image, crop, alignment, CODEX, and H&E helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import publication_context
from .utils import optional_import, save_figure_if_requested


def align_spatialdata_landmarks(
    sdata: Any,
    references_coords: Any,
    moving_coords: Any,
    reference_element: Any,
    moving_element: Any,
    reference_coordinate_system: str = "global",
    moving_coordinate_system: str = "global",
    new_coordinate_system: str = "aligned",
):
    """Align two SpatialData elements using landmarks and return the transform."""
    transformations = optional_import("spatialdata.transformations", "spatial")
    return transformations.align_elements_using_landmarks(
        references_coords=references_coords,
        moving_coords=moving_coords,
        reference_element=reference_element,
        moving_element=moving_element,
        reference_coordinate_system=reference_coordinate_system,
        moving_coordinate_system=moving_coordinate_system,
        new_coordinate_system=new_coordinate_system,
    )


def image_mask_to_labels(arr: Any, foreground_val: int | float, name: str = "mask_label"):
    """Convert a 2D mask value into a SpatialData labels object."""
    sd = optional_import("spatialdata", "spatial")
    models = optional_import("spatialdata.models", "spatial")
    labels = (np.asarray(arr) == foreground_val).astype("uint8")
    labels_elem = models.Labels2DModel.parse(labels)
    return sd.SpatialData(labels={name: labels_elem})


def binary_mask_to_spatialdata(arr: Any, name: str = "mask_binary"):
    """Convert a 2D binary mask into a SpatialData image object."""
    sd = optional_import("spatialdata", "spatial")
    models = optional_import("spatialdata.models", "spatial")
    image = models.Image2DModel.parse(np.asarray(arr).astype("uint8"))
    return sd.SpatialData(images={name: image})


def load_tiff_level(src: str | Path, series: int = 0, level: int = -1):
    """Load one level from a TIFF pyramid using tifffile."""
    tifffile = optional_import("tifffile", "spatial")
    with tifffile.TiffFile(str(src)) as tif:
        page_series = tif.series[series]
        if hasattr(page_series, "levels") and page_series.levels:
            return page_series.levels[level].asarray()
        return page_series.asarray()


def crop_spatialdata(
    sdata: Any,
    min_coordinate: Sequence[float],
    max_coordinate: Sequence[float],
    target_coordinate_system: str = "global",
):
    """Crop a SpatialData object by a bounding box."""
    spatialdata = optional_import("spatialdata", "spatial")
    return spatialdata.bounding_box_query(
        sdata,
        axes=("x", "y"),
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_coordinate_system=target_coordinate_system,
    )


def plot_spatial_crop(
    sdata: Any,
    labels_element: str = "cell_labels",
    image_element: str | None = None,
    color: str | None = None,
    table_name: str = "table",
    coordinate_systems: str = "aligned",
    figsize: tuple[float, float] = (8, 6),
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
):
    """Display a SpatialData crop with image and/or labels and return the plot object."""
    with publication_context(dpi=dpi):
        label_kwargs: dict[str, Any] = {"table_name": table_name}
        if color is not None:
            label_kwargs["color"] = color
        if image_element is None:
            plotter = sdata.pl.render_labels(element=labels_element, **label_kwargs)
        else:
            plotter = sdata.pl.render_images(element=image_element).pl.render_labels(
                element=labels_element,
                **label_kwargs,
            )
        result = plotter.pl.show(coordinate_systems=coordinate_systems, figsize=figsize)
        fig = plt.gcf()
        save_figure_if_requested(fig, save_path=save_path, dpi=dpi)
        if show:
            plt.show()
        return result


def plot_minmax_genes(
    sdata: Any,
    gene_list: Sequence[str],
    table_name: str = "table",
    labels_element: str = "cell_labels",
    coordinate_systems: str = "aligned",
    log1p: bool = True,
    contour_px: int = 3,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (5, 5),
    save_dir: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
) -> dict[str, Any]:
    """Plot min-max-normalized gene expression on SpatialData labels."""
    sc = optional_import("scanpy", "core")
    outputs: dict[str, Any] = {}
    table = sdata[table_name]
    for gene in gene_list:
        values = pd.to_numeric(sc.get.obs_df(table, keys=gene)[gene], errors="coerce").fillna(0)
        if log1p:
            values = np.log1p(values)
        denom = values.max() - values.min()
        table.obs[f"{gene}_expr_minmax"] = 0.0 if denom == 0 else (values - values.min()) / denom
        save_path = None
        if save_dir is not None:
            save_path = Path(save_dir) / f"{gene}.png"
        with publication_context(dpi=dpi):
            result = (
                sdata.pl.render_labels(
                    element=labels_element,
                    color=f"{gene}_expr_minmax",
                    table_name=table_name,
                    cmap=cmap,
                    contour_px=contour_px,
                )
                .pl.show(coordinate_systems=coordinate_systems, figsize=figsize, title=gene)
            )
            fig = plt.gcf()
            save_figure_if_requested(fig, save_path=save_path, dpi=dpi)
            if show:
                plt.show()
            outputs[gene] = result
    return outputs
