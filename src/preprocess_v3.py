from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


MAX_IMAGE_DIMENSION = 6000
MIN_IMAGE_DIMENSION = 8


def validate_image(image: Image.Image) -> None:
    """Validate an input image without modifying it."""

    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    width, height = image.size

    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ValueError("Image dimensions are too small.")

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError("Image dimensions exceed the V3 limit.")


def to_grayscale(image: Image.Image) -> np.ndarray:
    """Convert an image to an 8-bit grayscale array."""

    validate_image(image)

    grayscale = np.asarray(
        image.convert("L"),
        dtype=np.uint8,
    )

    if grayscale.ndim != 2 or grayscale.size == 0:
        raise ValueError("Unable to create a valid grayscale image.")

    return grayscale.copy()


def detect_background_polarity(
    grayscale: np.ndarray,
) -> dict[str, Any]:
    """
    Estimate background polarity from the image border.

    The result is a heuristic, not a guarantee for images with
    foreground objects touching most of their borders.
    """

    if not isinstance(grayscale, np.ndarray) or grayscale.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image.")

    if grayscale.size == 0:
        raise ValueError("Empty grayscale image.")

    border = np.concatenate([
        grayscale[0, :],
        grayscale[-1, :],
        grayscale[:, 0],
        grayscale[:, -1],
    ])

    border_median = float(np.median(border))

    return {
        "border_median": border_median,
        "background": (
            "light" if border_median >= 127.5 else "dark"
        ),
        "invert": border_median < 127.5,
    }


def enhance_contrast(grayscale: np.ndarray) -> np.ndarray:
    """Enhance local contrast while retaining the original dimensions."""

    if not isinstance(grayscale, np.ndarray) or grayscale.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image.")

    if grayscale.size == 0:
        raise ValueError("Empty grayscale image.")

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(grayscale)


def create_binary_image(
    enhanced: np.ndarray,
    invert: bool = False,
) -> np.ndarray:
    """
    Produce a binary image with white foreground and black background.

    Otsu thresholding is used initially. Adaptive thresholding is
    deliberately not substituted automatically because it can
    amplify paper texture and background noise.
    """

    if not isinstance(enhanced, np.ndarray) or enhanced.ndim != 2:
        raise ValueError("Expected a two-dimensional image.")

    if enhanced.size == 0:
        raise ValueError("Empty enhanced image.")

    working = enhanced

    if invert:
        working = cv2.bitwise_not(working)

    if np.min(working) == np.max(working):
        return np.zeros_like(working, dtype=np.uint8)

    _, binary = cv2.threshold(
        working,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    return binary


def orientation_diagnostics(binary: np.ndarray) -> dict[str, Any]:
    """
    Report dominant foreground orientation as a diagnostic only.

    PCA describes the dominant spatial axis of all foreground pixels.
    It does not reliably identify a writing baseline or individual
    digit rotation, so this estimate must not be used blindly.
    """

    if not isinstance(binary, np.ndarray) or binary.ndim != 2:
        raise ValueError("Expected a two-dimensional binary image.")

    ys, xs = np.where(binary > 0)

    if len(xs) < 10:
        return {
            "available": False,
            "dominant_axis_angle_degrees": None,
            "foreground_pixels": int(len(xs)),
            "warning": "Insufficient foreground for orientation analysis.",
        }

    points = np.column_stack((
        xs.astype(np.float32),
        ys.astype(np.float32),
    ))

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(
        points,
        mean=None,
    )

    vector = eigenvectors[0]
    angle = float(
        np.degrees(np.arctan2(vector[1], vector[0]))
    )

    angle = ((angle + 90.0) % 180.0) - 90.0

    return {
        "available": True,
        "dominant_axis_angle_degrees": angle,
        "foreground_pixels": int(len(xs)),
        "eigenvalues": [
            float(value) for value in eigenvalues.flatten()
        ],
        "warning": (
            "Dominant foreground axis is not necessarily "
            "the writing baseline or individual digit orientation."
        ),
    }


def preprocess_v3(image: Image.Image) -> dict[str, Any]:
    """
    Run the V3 preprocessing pipeline.

    Returns grayscale, enhanced grayscale, white-foreground binary
    image, polarity information and orientation diagnostics.
    """

    grayscale = to_grayscale(image)

    polarity = detect_background_polarity(grayscale)

    enhanced = enhance_contrast(grayscale)

    binary = create_binary_image(
        enhanced,
        invert=polarity["invert"],
    )

    foreground_ratio = float(np.count_nonzero(binary) / binary.size)

    diagnostics = orientation_diagnostics(binary)

    return {
        "grayscale": grayscale,
        "enhanced": enhanced,
        "binary": binary,
        "background": polarity["background"],
        "border_median": polarity["border_median"],
        "foreground_ratio": foreground_ratio,
        "orientation": diagnostics,
        "preprocessing_version": "v3",
    }
