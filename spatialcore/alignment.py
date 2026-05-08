"""Safe alignment and registration helpers for Xenium, CODEX, and H&E workflows.

The functions in this module operate on in-memory arrays, coordinates, and
shape-like objects. They do not read or write AnnData, SpatialData, images,
tables, or transformed outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from .plotting import publication_context
from .utils import optional_import, save_figure_if_requested

TransformLike = np.ndarray | Callable[[np.ndarray], np.ndarray]


def _as_coordinate_array(coords: Any, name: str) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"`{name}` must be an array-like object with shape `(n, 2)`.")
    return arr


def _as_affine_matrix(transform: Any) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix.shape != (3, 3):
        raise ValueError("Affine transforms must have shape `(3, 3)` or `(2, 3)`.")
    return matrix


def _minimum_landmarks(transform_type: str) -> int:
    if transform_type == "translation":
        return 1
    if transform_type in {"rigid", "similarity"}:
        return 2
    if transform_type == "affine":
        return 3
    raise ValueError("`transform_type` must be 'translation', 'rigid', 'similarity', or 'affine'.")


def _rms_error(reference_coords: np.ndarray, transformed_coords: np.ndarray) -> float:
    residual = reference_coords - transformed_coords
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def compute_affine_transform(
    reference_coords: Sequence[Sequence[float]],
    moving_coords: Sequence[Sequence[float]],
    transform_type: str = "affine",
) -> np.ndarray:
    """Estimate a transform that maps moving coordinates into reference coordinates.

    Supported landmark-based methods:

    - `translation`: centroid shift only.
    - `rigid`: rotation plus translation, no scaling.
    - `similarity`: rotation, uniform scaling, and translation.
    - `affine`: full 2D affine transform.

    Returns a 3 x 3 homogeneous matrix that maps `[x, y, 1]` from moving space
    into reference space.
    """
    transform_type = transform_type.lower()
    reference = _as_coordinate_array(reference_coords, "reference_coords")
    moving = _as_coordinate_array(moving_coords, "moving_coords")
    if reference.shape != moving.shape:
        raise ValueError("`reference_coords` and `moving_coords` must have the same shape.")

    min_points = _minimum_landmarks(transform_type)
    if reference.shape[0] < min_points:
        raise ValueError(f"`{transform_type}` alignment requires at least {min_points} landmarks.")

    if transform_type == "translation":
        offset = reference.mean(axis=0) - moving.mean(axis=0)
        return np.array(
            [
                [1.0, 0.0, offset[0]],
                [0.0, 1.0, offset[1]],
                [0.0, 0.0, 1.0],
            ]
        )

    if transform_type in {"rigid", "similarity"}:
        moving_centroid = moving.mean(axis=0)
        reference_centroid = reference.mean(axis=0)
        moving_centered = moving - moving_centroid
        reference_centered = reference - reference_centroid
        covariance = moving_centered.T @ reference_centered
        u, singular_values, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = vt.T @ u.T
        scale = 1.0
        if transform_type == "similarity":
            denom = np.sum(moving_centered**2)
            if denom == 0:
                raise ValueError("Cannot estimate similarity scale from collapsed landmarks.")
            scale = float(np.sum(singular_values) / denom)
        linear = scale * rotation
        translation = reference_centroid - linear @ moving_centroid
        return np.array(
            [
                [linear[0, 0], linear[0, 1], translation[0]],
                [linear[1, 0], linear[1, 1], translation[1]],
                [0.0, 0.0, 1.0],
            ]
        )

    design = np.column_stack([moving, np.ones(moving.shape[0])])
    coefficients, *_ = np.linalg.lstsq(design, reference, rcond=None)
    return np.array(
        [
            [coefficients[0, 0], coefficients[1, 0], coefficients[2, 0]],
            [coefficients[0, 1], coefficients[1, 1], coefficients[2, 1]],
            [0.0, 0.0, 1.0],
        ]
    )


def apply_transform_to_coordinates(
    coords: Sequence[Sequence[float]],
    transform: TransformLike,
    inverse: bool = False,
) -> np.ndarray:
    """Apply an affine matrix or callable nonlinear transform to 2D coordinates.

    Callable transforms are accepted for future nonlinear/deformable workflows,
    but this package does not yet estimate nonlinear transforms itself.
    """
    arr = _as_coordinate_array(coords, "coords")
    if callable(transform):
        if inverse:
            raise NotImplementedError("Inverse callable transforms are not supported.")
        transformed = np.asarray(transform(arr), dtype=float)
        return _as_coordinate_array(transformed, "transformed coordinates")

    matrix = _as_affine_matrix(transform)
    if inverse:
        matrix = np.linalg.inv(matrix)
    hom = np.column_stack([arr, np.ones(arr.shape[0])])
    return (hom @ matrix.T)[:, :2]


def apply_transform_to_shapes(shapes: Any, transform: Any, inverse: bool = False) -> Any:
    """Apply an affine transform to shapely/geopandas geometry-like objects.

    Supported inputs are GeoDataFrames, GeoSeries, lists/tuples of shapely
    geometries, or a single shapely geometry. Nonlinear shape warping is not
    implemented because it requires a concrete deformation model.
    """
    if callable(transform):
        raise NotImplementedError("Callable/nonlinear transforms are not supported for shapes yet.")
    matrix = _as_affine_matrix(transform)
    if inverse:
        matrix = np.linalg.inv(matrix)
    shapely_affinity = optional_import("shapely.affinity", "codex-he")
    params = [matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]]

    def _transform_geom(geom: Any) -> Any:
        return shapely_affinity.affine_transform(geom, params)

    if hasattr(shapes, "geometry") and hasattr(shapes, "columns"):
        out = shapes.copy()
        out["geometry"] = out.geometry.apply(_transform_geom)
        return out
    if hasattr(shapes, "apply"):
        return shapes.apply(_transform_geom)
    if isinstance(shapes, list):
        return [_transform_geom(geom) for geom in shapes]
    if isinstance(shapes, tuple):
        return tuple(_transform_geom(geom) for geom in shapes)
    return _transform_geom(shapes)


def register_image_pair(
    reference_image: Any | None = None,
    moving_image: Any | None = None,
    method: str = "landmark",
    reference_landmarks: Sequence[Sequence[float]] | None = None,
    moving_landmarks: Sequence[Sequence[float]] | None = None,
    transform_type: str = "affine",
    **kwargs: Any,
) -> dict[str, Any]:
    """Register an image pair or landmark pair without reading or writing files.

    Supported methods:

    - `landmark` or `manual`: use supplied paired landmarks and
      `compute_affine_transform`.
    - `translation` or `phase_cross_correlation`: estimate translation using
      `skimage.registration.phase_cross_correlation` when scikit-image is
      installed.

    Nonlinear/deformable registration is intentionally a placeholder for now.
    """
    method = method.lower()
    if method in {"landmark", "manual", "landmarks"}:
        if reference_landmarks is None or moving_landmarks is None:
            raise ValueError("Landmark registration requires reference and moving landmarks.")
        transform = compute_affine_transform(
            reference_coords=reference_landmarks,
            moving_coords=moving_landmarks,
            transform_type=transform_type,
        )
        reference = _as_coordinate_array(reference_landmarks, "reference_landmarks")
        moving = _as_coordinate_array(moving_landmarks, "moving_landmarks")
        transformed = apply_transform_to_coordinates(moving, transform)
        return {
            "method": "landmark",
            "transform_type": transform_type,
            "transform": transform,
            "rms_error": _rms_error(reference, transformed),
            "moving_landmarks_transformed": transformed,
        }

    if method in {"translation", "phase_cross_correlation"}:
        if reference_image is None or moving_image is None:
            raise ValueError("Translation image registration requires reference and moving images.")
        registration = optional_import("skimage.registration", "codex-he")
        reference = _to_grayscale(reference_image)
        moving = _to_grayscale(moving_image)
        shift, error, phasediff = registration.phase_cross_correlation(
            reference,
            moving,
            **kwargs,
        )
        row_shift, col_shift = np.asarray(shift, dtype=float)[:2]
        transform = np.array(
            [
                [1.0, 0.0, col_shift],
                [0.0, 1.0, row_shift],
                [0.0, 0.0, 1.0],
            ]
        )
        return {
            "method": "phase_cross_correlation",
            "transform_type": "translation",
            "transform": transform,
            "shift_yx": (float(row_shift), float(col_shift)),
            "error": float(error),
            "phasediff": float(phasediff),
        }

    if method in {"deformable", "nonlinear", "elastic"}:
        raise NotImplementedError(
            "Nonlinear/deformable registration is not implemented yet. "
            "Planned options include SimpleITK/ANTs-style backends with explicit "
            "user opt-in and synthetic tests."
        )
    raise ValueError("`method` must be 'landmark', 'translation', or 'deformable'.")


def align_xenium_to_he(
    xenium_landmarks: Sequence[Sequence[float]],
    he_landmarks: Sequence[Sequence[float]],
    xenium_coordinates: Sequence[Sequence[float]] | None = None,
    transform_type: str = "affine",
) -> dict[str, Any]:
    """Estimate a landmark transform from Xenium coordinates into H&E space.

    This is a safe in-memory helper. It does not load Xenium or H&E files and
    does not write transformed outputs.
    """
    result = register_image_pair(
        method="landmark",
        reference_landmarks=he_landmarks,
        moving_landmarks=xenium_landmarks,
        transform_type=transform_type,
    )
    result["moving_modality"] = "xenium"
    result["reference_modality"] = "he"
    if xenium_coordinates is not None:
        result["transformed_coordinates"] = apply_transform_to_coordinates(
            xenium_coordinates,
            result["transform"],
        )
    return result


def align_xenium_to_codex(
    xenium_landmarks: Sequence[Sequence[float]],
    codex_landmarks: Sequence[Sequence[float]],
    xenium_coordinates: Sequence[Sequence[float]] | None = None,
    transform_type: str = "affine",
) -> dict[str, Any]:
    """Estimate a landmark transform from Xenium coordinates into CODEX space.

    This is a safe in-memory helper. It does not load Xenium or CODEX files and
    does not write transformed outputs.
    """
    result = register_image_pair(
        method="landmark",
        reference_landmarks=codex_landmarks,
        moving_landmarks=xenium_landmarks,
        transform_type=transform_type,
    )
    result["moving_modality"] = "xenium"
    result["reference_modality"] = "codex"
    if xenium_coordinates is not None:
        result["transformed_coordinates"] = apply_transform_to_coordinates(
            xenium_coordinates,
            result["transform"],
        )
    return result


def _to_grayscale(image: Any) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr.mean(axis=-1)
    raise ValueError("Images must be 2D grayscale or 3D channel-last arrays.")


def _normalize_image(image: Any) -> np.ndarray:
    arr = _to_grayscale(image)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    vmin, vmax = np.percentile(finite, [1, 99])
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1)


def _warp_moving_to_reference(
    moving_image: Any,
    transform: Any,
    output_shape: tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    matrix = np.linalg.inv(_as_affine_matrix(transform))
    row_col_matrix = np.array(
        [
            [matrix[1, 1], matrix[1, 0]],
            [matrix[0, 1], matrix[0, 0]],
        ]
    )
    offset = np.array([matrix[1, 2], matrix[0, 2]])
    moving = np.asarray(moving_image, dtype=float)
    if moving.ndim == 2:
        return ndimage.affine_transform(
            moving,
            row_col_matrix,
            offset=offset,
            output_shape=output_shape,
            order=order,
            mode="constant",
            cval=0,
        )
    if moving.ndim == 3:
        channels = [
            ndimage.affine_transform(
                moving[..., channel],
                row_col_matrix,
                offset=offset,
                output_shape=output_shape,
                order=order,
                mode="constant",
                cval=0,
            )
            for channel in range(moving.shape[-1])
        ]
        return np.stack(channels, axis=-1)
    raise ValueError("Moving image must be 2D grayscale or 3D channel-last.")


def overlay_images(
    reference_image: Any,
    moving_image: Any,
    transform: Any | None = None,
    alpha: float = 0.55,
    moving_order: int = 1,
) -> np.ndarray:
    """Create an RGB overlay for alignment QC without saving a plot.

    The reference image is shown in red and the moving image in green. If an
    affine transform is supplied, the moving image is warped into the reference
    image grid before overlaying.
    """
    reference = _to_grayscale(reference_image)
    moving = moving_image
    if transform is not None:
        moving = _warp_moving_to_reference(
            moving_image=moving_image,
            transform=transform,
            output_shape=reference.shape,
            order=moving_order,
        )
    moving_gray = _to_grayscale(moving)
    if reference.shape != moving_gray.shape:
        raise ValueError("Images must have the same shape unless an affine transform is supplied.")

    ref_norm = _normalize_image(reference)
    mov_norm = _normalize_image(moving_gray)
    overlay = np.zeros((*reference.shape, 3), dtype=float)
    overlay[..., 0] = alpha * ref_norm
    overlay[..., 1] = alpha * mov_norm
    overlay[..., 2] = 0.25 * (ref_norm + mov_norm)
    return np.clip(overlay, 0, 1)


def plot_alignment_qc(
    reference_image: Any | None = None,
    moving_image: Any | None = None,
    transform: Any | None = None,
    reference_coords: Sequence[Sequence[float]] | None = None,
    moving_coords: Sequence[Sequence[float]] | None = None,
    transformed_coords: Sequence[Sequence[float]] | None = None,
    title: str = "Alignment QC",
    figsize: tuple[float, float] = (4, 4),
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
):
    """Plot image overlay and/or landmark QC for an alignment result.

    The plot is displayed by default, returns `fig, ax`, and saves only when
    `save_path` is explicitly provided.
    """
    with publication_context(dpi=dpi):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if reference_image is not None and moving_image is not None:
            ax.imshow(overlay_images(reference_image, moving_image, transform=transform))
            ax.set_axis_off()

        if reference_coords is not None:
            reference = _as_coordinate_array(reference_coords, "reference_coords")
            ax.scatter(reference[:, 0], reference[:, 1], s=24, c="#d62728", label="reference")
        if moving_coords is not None:
            moving = _as_coordinate_array(moving_coords, "moving_coords")
            if transformed_coords is None and transform is not None:
                transformed_coords = apply_transform_to_coordinates(moving, transform)
            if transformed_coords is None:
                ax.scatter(moving[:, 0], moving[:, 1], s=18, c="#1f77b4", label="moving")
            else:
                transformed = _as_coordinate_array(transformed_coords, "transformed_coords")
                ax.scatter(
                    transformed[:, 0],
                    transformed[:, 1],
                    s=18,
                    c="#2ca02c",
                    marker="x",
                    label="moving transformed",
                )
                if reference_coords is not None and transformed.shape == reference.shape:
                    for ref, mov in zip(reference, transformed):
                        ax.plot([ref[0], mov[0]], [ref[1], mov[1]], color="0.6", lw=0.6)

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        if reference_coords is not None or moving_coords is not None:
            ax.legend(frameon=False, loc="best")
        save_figure_if_requested(fig, save_path=save_path, dpi=dpi)
        if show:
            plt.show()
        return fig, ax
