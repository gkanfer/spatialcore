from __future__ import annotations

import numpy as np
import pytest

from spatialcore.alignment import (
    align_xenium_to_he,
    apply_transform_to_coordinates,
    compute_affine_transform,
    overlay_images,
    register_image_pair,
)


def test_compute_affine_transform_maps_moving_to_reference():
    moving = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    reference = np.array([[2, 3], [4, 3], [2, 5], [4, 5]], dtype=float)

    transform = compute_affine_transform(reference, moving, transform_type="affine")
    transformed = apply_transform_to_coordinates(moving, transform)

    np.testing.assert_allclose(transformed, reference)


def test_align_xenium_to_he_returns_transform_and_coordinates():
    xenium = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    he = np.array([[5, 5], [6, 5], [5, 6]], dtype=float)
    coords = np.array([[2, 2]], dtype=float)

    result = align_xenium_to_he(
        xenium_landmarks=xenium,
        he_landmarks=he,
        xenium_coordinates=coords,
        transform_type="affine",
    )

    assert result["moving_modality"] == "xenium"
    assert result["reference_modality"] == "he"
    np.testing.assert_allclose(result["transformed_coordinates"], [[7, 7]])


def test_overlay_images_returns_rgb_array():
    reference = np.zeros((8, 8), dtype=float)
    moving = np.zeros((8, 8), dtype=float)
    reference[2:5, 2:5] = 1
    moving[3:6, 3:6] = 1

    overlay = overlay_images(reference, moving)

    assert overlay.shape == (8, 8, 3)
    assert overlay.max() <= 1
    assert overlay.min() >= 0


def test_deformable_registration_is_explicit_placeholder():
    with pytest.raises(NotImplementedError, match="Nonlinear/deformable"):
        register_image_pair(method="deformable")
