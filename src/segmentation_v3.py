from __future__ import annotations

from typing import Any

import cv2
import numpy as np


MIN_COMPONENT_AREA = 3
MIN_CANDIDATE_WIDTH = 5
MIN_VALLEY_DEPTH = 0.35
MAX_SPLITS_PER_COMPONENT = 8


def extract_components_v3(
    binary: np.ndarray,
) -> list[dict[str, Any]]:
    """Extract foreground components while retaining small components."""

    if not isinstance(binary, np.ndarray) or binary.ndim != 2:
        raise ValueError("Expected a two-dimensional binary image.")

    if binary.size == 0:
        raise ValueError("Empty binary image.")

    foreground = np.ascontiguousarray(
        (binary > 0).astype(np.uint8)
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )

    components = []

    for index in range(1, count):
        x, y, width, height, area = stats[index]

        if area < MIN_COMPONENT_AREA:
            continue

        components.append({
            "id": int(index),
            "bbox": [
                int(x), int(y), int(width), int(height)
            ],
            "area": int(area),
            "centroid": [
                float(centroids[index][0]),
                float(centroids[index][1]),
            ],
            "image": foreground[
                y:y + height,
                x:x + width,
            ].copy() * 255,
        })

    components.sort(
        key=lambda item: (
            item["bbox"][0],
            item["bbox"][1],
        )
    )

    return components


def find_vertical_split_candidates(
    component: np.ndarray,
    max_splits: int = MAX_SPLITS_PER_COMPONENT,
) -> list[dict[str, Any]]:
    """
    Find possible vertical boundaries using column projections.

    Candidates are hypotheses only. They are not guaranteed to
    correspond to actual boundaries between handwritten digits.
    """

    if not isinstance(component, np.ndarray) or component.ndim != 2:
        raise ValueError("Expected a two-dimensional component.")

    if component.size == 0:
        raise ValueError("Empty component.")

    if max_splits < 1:
        raise ValueError("max_splits must be positive.")

    binary = (component > 0).astype(np.uint8)

    height, width = binary.shape

    if width < MIN_CANDIDATE_WIDTH * 2:
        return []

    projection = binary.sum(axis=0).astype(np.float32)

    smoothed = np.convolve(
        projection,
        np.ones(3, dtype=np.float32) / 3.0,
        mode="same",
    )

    peak = float(smoothed.max())

    if peak <= 0:
        return []

    candidates = []

    for x in range(MIN_CANDIDATE_WIDTH, width - MIN_CANDIDATE_WIDTH):
        left_peak = float(smoothed[:x].max())
        right_peak = float(smoothed[x + 1:].max())

        local_reference = min(left_peak, right_peak)

        if local_reference <= 0:
            continue

        depth = 1.0 - float(smoothed[x] / local_reference)

        if depth < MIN_VALLEY_DEPTH:
            continue

        candidates.append({
            "x": int(x),
            "valley_pixels": int(projection[x]),
            "valley_depth": float(depth),
            "normalized_ink": float(
                projection[x] / max(height, 1)
            ),
        })

    candidates.sort(
        key=lambda item: (
            -item["valley_depth"],
            item["valley_pixels"],
            item["x"],
        )
    )

    selected = []

    for candidate in candidates:
        if any(
            abs(candidate["x"] - existing["x"]) < 7
            for existing in selected
        ):
            continue

        selected.append(candidate)

        if len(selected) >= max_splits:
            break

    return sorted(selected, key=lambda item: item["x"])


def create_split_hypotheses(
    component: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Create two-part crop hypotheses for candidate split positions.

    The original component is never discarded. Every split remains
    an alternative requiring recognition-level evaluation.
    """

    image = component.get("image")

    if not isinstance(image, np.ndarray):
        raise ValueError("Component must contain a NumPy image.")

    if image.ndim != 2 or image.size == 0:
        raise ValueError("Invalid component image.")

    candidates = find_vertical_split_candidates(image)

    hypotheses = []

    for candidate in candidates:
        x = candidate["x"]

        left = image[:, :x].copy()
        right = image[:, x:].copy()

        if not np.any(left) or not np.any(right):
            continue

        hypotheses.append({
            "split_x": x,
            "left": left,
            "right": right,
            "valley_depth": candidate["valley_depth"],
            "valley_pixels": candidate["valley_pixels"],
        })

    return hypotheses


def segment_v3(binary: np.ndarray) -> dict[str, Any]:
    """
    Extract components and generate possible split boundaries.

    This stage does not claim that components equal digits.
    """

    components = extract_components_v3(binary)

    results = []

    for component in components:
        width = component["bbox"][2]
        height = component["bbox"][3]

        hypotheses = create_split_hypotheses(component)

        results.append({
            **component,
            "aspect_ratio": float(width / max(height, 1)),
            "split_candidates": hypotheses,
            "possible_multi_digit_component": bool(hypotheses),
        })

    return {
        "components": results,
        "component_count": len(results),
        "candidate_split_count": sum(
            len(item["split_candidates"])
            for item in results
        ),
        "segmentation_version": "v3",
        "warning": (
            "Connected components and projection valleys are "
            "segmentation hypotheses, not verified digit boundaries."
        ),
    }
