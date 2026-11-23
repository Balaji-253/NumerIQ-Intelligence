from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


MAX_IMAGE_DIMENSION = 3000
MIN_COMPONENT_AREA = 8
MIN_COMPONENT_HEIGHT_RATIO = 0.08
BOUNDING_BOX_PADDING = 4


def validate_image(image: Image.Image) -> None:
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    if image.width <= 0 or image.height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if (
        image.width > MAX_IMAGE_DIMENSION
        or image.height > MAX_IMAGE_DIMENSION
    ):
        raise ValueError("Image dimensions exceed the 3000-pixel limit.")


def to_grayscale(image: Image.Image) -> np.ndarray:
    """Convert PIL image into a grayscale NumPy array."""
    validate_image(image)

    image = image.convert("RGB")
    array = np.asarray(image)

    gray = cv2.cvtColor(
        array,
        cv2.COLOR_RGB2GRAY,
    )

    return gray


def resize_image(gray: np.ndarray) -> np.ndarray:
    """Resize oversized images while preserving aspect ratio."""
    height, width = gray.shape

    largest_dimension = max(height, width)

    if largest_dimension <= MAX_IMAGE_DIMENSION:
        return gray

    scale = MAX_IMAGE_DIMENSION / largest_dimension

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        gray,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def enhance_image(image: Image.Image) -> dict[str, Any]:
    """
    Enhance a handwritten image.

    Returns:
        grayscale: denoised grayscale image
        enhanced: contrast-enhanced grayscale image
        binary: white foreground on black background
    """
    gray = to_grayscale(image)
    gray = resize_image(gray)

    denoised = cv2.medianBlur(gray, 3)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(denoised)

    threshold, binary = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # A uniform image contains no handwriting, regardless of its color.
    if int(enhanced.max()) == int(enhanced.min()):
        binary = np.zeros_like(enhanced)
    else:
        # Determine whether the background is predominantly white.
        border = np.concatenate([
            enhanced[0, :],
            enhanced[-1, :],
            enhanced[:, 0],
            enhanced[:, -1],
        ])

        border_mean = float(np.mean(border))
        image_mean = float(np.mean(enhanced))

        # Convert to white foreground on black background.
        if border_mean > image_mean:
            binary = cv2.bitwise_not(binary)

    return {
        "grayscale": denoised,
        "enhanced": enhanced,
        "binary": binary,
        "threshold": float(threshold),
    }


def _extract_components(
    binary: np.ndarray,
) -> list[tuple[int, int, int, int, int]]:
    """Extract connected components and filter obvious noise."""
    height, width = binary.shape

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    components = []

    minimum_height = max(
        5,
        int(height * MIN_COMPONENT_HEIGHT_RATIO),
    )

    for index in range(1, count):
        x, y, w, h, area = stats[index]

        if area < MIN_COMPONENT_AREA:
            continue

        if h < minimum_height:
            continue

        if w < 2:
            continue

        if w > width * 0.95:
            continue

        components.append((
            int(x),
            int(y),
            int(w),
            int(h),
            int(area),
        ))

    return components


def segment_digits(
    image: Image.Image,
) -> list[dict[str, Any]]:
    """
    Locate separated foreground components from left to right.

    Each result contains:
        bbox: [x, y, width, height]
        image: cropped binary NumPy image
        area: foreground pixel count

    Important:
        Connected components are not guaranteed to equal digits.
        Touching digits and disconnected strokes require additional
        recognition-aware segmentation.
    """
    processed = enhance_image(image)

    binary = processed["binary"]

    components = _extract_components(binary)

    components.sort(
        key=lambda component: (
            component[0],
            component[1],
        )
    )

    height, width = binary.shape

    results = []

    for x, y, w, h, area in components:
        left = max(0, x - BOUNDING_BOX_PADDING)
        top = max(0, y - BOUNDING_BOX_PADDING)

        right = min(
            width,
            x + w + BOUNDING_BOX_PADDING,
        )

        bottom = min(
            height,
            y + h + BOUNDING_BOX_PADDING,
        )

        crop = binary[top:bottom, left:right].copy()

        results.append({
            "bbox": [
                int(left),
                int(top),
                int(right - left),
                int(bottom - top),
            ],
            "image": crop,
            "area": int(area),
        })

    return results
