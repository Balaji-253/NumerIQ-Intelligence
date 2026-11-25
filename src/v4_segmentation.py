from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class V4SegmentationConfig:
    min_component_area: int = 3
    min_split_edge_margin: int = 4
    min_split_separation: int = 5
    min_valley_depth: float = 0.25
    projection_smoothing_kernel: int = 3
    max_candidates_per_component: int = 12


@dataclass
class SplitCandidate:
    x: int
    valley_depth: float
    valley_ratio: float
    left_mass: int
    right_mass: int


@dataclass
class ComponentAnalysis:
    component_id: int
    bbox: tuple[int, int, int, int]
    area: int
    aspect_ratio: float
    centroid: tuple[float, float]
    is_noise: bool
    split_candidates: list[SplitCandidate]


@dataclass
class SegmentationAnalysis:
    status: str
    image_width: int
    image_height: int
    component_count: int
    noise_component_count: int
    orientation_degrees: float | None
    components: list[ComponentAnalysis]
    segmentation_version: str = "v4.0"


DEFAULT_CONFIG = V4SegmentationConfig()


def validate_binary_image(binary: np.ndarray) -> np.ndarray:
    """Validate and normalize a binary image with white foreground."""
    if not isinstance(binary, np.ndarray):
        raise TypeError("binary must be a NumPy array")

    if binary.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image")

    if binary.size == 0:
        raise ValueError("Image cannot be empty")

    if binary.shape[0] < 1 or binary.shape[1] < 1:
        raise ValueError("Image dimensions must be positive")

    return np.where(binary > 0, 255, 0).astype(np.uint8)


def estimate_orientation(binary: np.ndarray) -> float | None:
    """
    Estimate the dominant foreground orientation using PCA.

    This is diagnostic only. It does not rotate the input image.
    """
    points = np.column_stack(np.where(binary > 0))

    if len(points) < 3:
        return None

    coordinates = points[:, ::-1].astype(np.float64)
    centered = coordinates - coordinates.mean(axis=0)

    covariance = np.cov(centered, rowvar=False)

    if not np.all(np.isfinite(covariance)):
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    angle = np.degrees(
        np.arctan2(principal[1], principal[0])
    )

    # Normalize the undirected dominant axis to [-90, 90].
    angle = ((angle + 90.0) % 180.0) - 90.0

    return float(angle)


def _smooth_projection(
    projection: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones(kernel_size, dtype=np.float32)
    kernel /= kernel.sum()

    return np.convolve(
        projection.astype(np.float32),
        kernel,
        mode="same",
    )


def find_split_candidates(
    component_mask: np.ndarray,
    config: V4SegmentationConfig = DEFAULT_CONFIG,
) -> list[SplitCandidate]:
    """
    Find candidate vertical cuts using foreground column projections.

    These are hypotheses only. No split is automatically accepted.
    """
    if component_mask.ndim != 2:
        raise ValueError("Component mask must be two-dimensional")

    height, width = component_mask.shape

    if width < 2 * config.min_split_edge_margin + 1:
        return []

    projection = np.count_nonzero(
        component_mask > 0,
        axis=0,
    )

    smoothed = _smooth_projection(
        projection,
        config.projection_smoothing_kernel,
    )

    candidates: list[SplitCandidate] = []

    start = config.min_split_edge_margin
    end = width - config.min_split_edge_margin

    for x in range(start, end):
        if smoothed[x] > smoothed[x - 1]:
            continue

        if smoothed[x] > smoothed[x + 1]:
            continue

        # Use a width-adaptive neighborhood so thin bridges
        # can be compared against the wider foreground regions.
        radius = max(4, width // 8)
        left_start = max(0, x - radius)
        right_end = min(width, x + radius + 1)

        left_peak = float(np.max(smoothed[left_start:x]))
        right_peak = float(np.max(smoothed[x + 1:right_end]))

        shoulder = min(left_peak, right_peak)

        if shoulder <= 0:
            continue

        valley_ratio = float(smoothed[x] / shoulder)
        valley_depth = 1.0 - valley_ratio

        if valley_depth < config.min_valley_depth:
            continue

        left_mass = int(np.count_nonzero(component_mask[:, :x]))
        right_mass = int(np.count_nonzero(component_mask[:, x:]))

        if left_mass == 0 or right_mass == 0:
            continue

        candidate = SplitCandidate(
            x=x,
            valley_depth=float(valley_depth),
            valley_ratio=valley_ratio,
            left_mass=left_mass,
            right_mass=right_mass,
        )

        candidates.append(candidate)

    # Keep separated local candidates, preferring deeper valleys.
    candidates.sort(
        key=lambda item: item.valley_depth,
        reverse=True,
    )

    selected: list[SplitCandidate] = []

    for candidate in candidates:
        if any(
            abs(candidate.x - existing.x)
            < config.min_split_separation
            for existing in selected
        ):
            continue

        selected.append(candidate)

        if len(selected) >= config.max_candidates_per_component:
            break

    return sorted(selected, key=lambda item: item.x)


def extract_components(
    binary: np.ndarray,
    config: V4SegmentationConfig = DEFAULT_CONFIG,
) -> list[ComponentAnalysis]:
    """Extract connected components and preserve their geometry."""
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    components: list[ComponentAnalysis] = []

    for label in range(1, count):
        x, y, width, height, area = (
            int(value) for value in stats[label]
        )

        component_mask = np.where(
            labels[y:y + height, x:x + width] == label,
            255,
            0,
        ).astype(np.uint8)

        is_noise = area < config.min_component_area

        candidates = (
            []
            if is_noise
            else find_split_candidates(component_mask, config)
        )

        components.append(
            ComponentAnalysis(
                component_id=label,
                bbox=(x, y, width, height),
                area=area,
                aspect_ratio=float(width / max(height, 1)),
                centroid=(
                    float(centroids[label][0]),
                    float(centroids[label][1]),
                ),
                is_noise=is_noise,
                split_candidates=candidates,
            )
        )

    # Stable reading-order arrangement.
    components.sort(
        key=lambda item: (
            item.bbox[0],
            item.bbox[1],
        )
    )

    return components


def analyze_binary_image(
    binary: np.ndarray,
    config: V4SegmentationConfig = DEFAULT_CONFIG,
) -> SegmentationAnalysis:
    """
    Analyze a binary image without making definitive digit-count claims.

    Input convention:
        White foreground (255)
        Black background (0)
    """
    normalized = validate_binary_image(binary)
    height, width = normalized.shape

    components = extract_components(normalized, config)

    noise_count = sum(
        component.is_noise for component in components
    )

    valid_count = len(components) - noise_count

    return SegmentationAnalysis(
        status=(
            "no_foreground"
            if valid_count == 0
            else "analysis_completed"
        ),
        image_width=width,
        image_height=height,
        component_count=len(components),
        noise_component_count=noise_count,
        orientation_degrees=estimate_orientation(normalized),
        components=components,
    )


def analysis_to_dict(
    analysis: SegmentationAnalysis,
) -> dict[str, Any]:
    """Convert analysis results into JSON-compatible data."""
    return asdict(analysis)
