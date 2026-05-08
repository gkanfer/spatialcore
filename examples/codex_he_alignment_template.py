"""Template for CODEX/H&E SpatialData alignment.

This file documents the intended API and does not run automatically.
"""

from spatialcore.alignment import (
    align_xenium_to_codex,
    align_xenium_to_he,
    plot_alignment_qc,
)
from spatialcore.image import align_spatialdata_landmarks, crop_spatialdata, plot_spatial_crop
from spatialcore.io import read_spatialdata


def codex_he_alignment_template(spatialdata_path, min_coordinate, max_coordinate):
    sdata = read_spatialdata(spatialdata_path)
    cropped = crop_spatialdata(sdata, min_coordinate=min_coordinate, max_coordinate=max_coordinate)
    plot_spatial_crop(cropped, save_path=None)
    return cropped


def landmark_alignment_template(
    sdata,
    reference_coords,
    moving_coords,
    reference_element,
    moving_element,
):
    return align_spatialdata_landmarks(
        sdata=sdata,
        references_coords=reference_coords,
        moving_coords=moving_coords,
        reference_element=reference_element,
        moving_element=moving_element,
    )


def xenium_to_he_landmark_template(xenium_landmarks, he_landmarks, xenium_coordinates=None):
    result = align_xenium_to_he(
        xenium_landmarks=xenium_landmarks,
        he_landmarks=he_landmarks,
        xenium_coordinates=xenium_coordinates,
        transform_type="affine",
    )
    plot_alignment_qc(
        reference_coords=he_landmarks,
        moving_coords=xenium_landmarks,
        transform=result["transform"],
        save_path=None,
    )
    return result


def xenium_to_codex_landmark_template(xenium_landmarks, codex_landmarks, xenium_coordinates=None):
    result = align_xenium_to_codex(
        xenium_landmarks=xenium_landmarks,
        codex_landmarks=codex_landmarks,
        xenium_coordinates=xenium_coordinates,
        transform_type="affine",
    )
    plot_alignment_qc(
        reference_coords=codex_landmarks,
        moving_coords=xenium_landmarks,
        transform=result["transform"],
        save_path=None,
    )
    return result
