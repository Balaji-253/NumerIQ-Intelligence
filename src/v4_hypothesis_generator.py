from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np

from src.v4_segmentation import (
    DEFAULT_CONFIG,
    V4SegmentationConfig,
    find_split_candidates,
    validate_binary_image,
)


@dataclass
class HypothesisPart:
    part_index: int
    bbox: tuple[int, int, int, int]
    area: int
    foreground: np.ndarray


@dataclass
class SegmentationHypothesis:
    hypothesis_id: str
    component_id: int
    strategy: str
    cut_positions: tuple[int, ...]
    parts: list[HypothesisPart]
    geometry_score: float
    is_original: bool
    automatically_accepted: bool = False


@dataclass
class ComponentHypothesisSet:
    component_id: int
    original_bbox: tuple[int, int, int, int]
    hypotheses: list[SegmentationHypothesis]
    image_bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class HypothesisConfig:
    max_cuts_per_component: int = 2
    min_part_width: int = 3
    min_part_area: int = 3
    max_hypotheses_per_component: int = 50


def _make_parts(
    component: np.ndarray,
    cut_positions: tuple[int, ...],
    min_width: int,
    min_area: int,
) -> list[HypothesisPart] | None:
    """Create nonempty foreground regions from vertical cut positions."""
    height, width = component.shape

    boundaries = (0, *cut_positions, width)
    parts: list[HypothesisPart] = []

    for index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        if end - start < min_width:
            return None

        crop = component[:, start:end].copy()

        foreground_y, foreground_x = np.where(crop > 0)
        area = int(len(foreground_x))

        if area < min_area:
            return None

        x0 = int(foreground_x.min())
        x1 = int(foreground_x.max()) + 1
        y0 = int(foreground_y.min())
        y1 = int(foreground_y.max()) + 1

        tight_crop = crop[y0:y1, x0:x1].copy()

        parts.append(
            HypothesisPart(
                part_index=index,
                bbox=(
                    start + x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                ),
                area=area,
                foreground=tight_crop,
            )
        )

    return parts


def generate_component_hypotheses(
    component: np.ndarray,
    component_id: int,
    config: HypothesisConfig = HypothesisConfig(),
    segmentation_config: V4SegmentationConfig = DEFAULT_CONFIG,
    image_bbox: tuple[int, int, int, int] | None = None,
) -> ComponentHypothesisSet:
    """
    Generate original, single-cut and multiple-cut interpretations.

    Coordinates in cut_positions are relative to the component.
    Part bounding boxes are also relative to the component.

    These hypotheses are alternatives, not verified digit boundaries.
    """
    component = validate_binary_image(component)

    if component_id < 1:
        raise ValueError("component_id must be a positive integer")

    if config.max_cuts_per_component < 0:
        raise ValueError("max_cuts_per_component cannot be negative")

    if config.min_part_width < 1 or config.min_part_area < 1:
        raise ValueError("Minimum part dimensions and area must be positive")

    if config.max_hypotheses_per_component < 1:
        raise ValueError("max_hypotheses_per_component must be positive")

    foreground_y, foreground_x = np.where(component > 0)

    if len(foreground_x) == 0:
        return ComponentHypothesisSet(
            component_id=component_id,
            original_bbox=(0, 0, 0, 0),
            hypotheses=[],
            image_bbox=image_bbox,
        )

    original_bbox = (
        int(foreground_x.min()),
        int(foreground_y.min()),
        int(foreground_x.max() - foreground_x.min() + 1),
        int(foreground_y.max() - foreground_y.min() + 1),
    )

    hypotheses: list[SegmentationHypothesis] = []

    # Always preserve the unsplit original component.
    original_parts = _make_parts(
        component,
        (),
        config.min_part_width,
        config.min_part_area,
    )

    if original_parts is not None:
        hypotheses.append(
            SegmentationHypothesis(
                hypothesis_id=f"c{component_id}_original",
                component_id=component_id,
                strategy="original",
                cut_positions=(),
                parts=original_parts,
                geometry_score=0.0,
                is_original=True,
            )
        )

    if config.max_cuts_per_component == 0:
        return ComponentHypothesisSet(
            component_id=component_id,
            original_bbox=original_bbox,
            hypotheses=hypotheses,
            image_bbox=image_bbox,
        )

    candidates = find_split_candidates(
        component,
        segmentation_config,
    )

    candidate_by_x = {
        candidate.x: candidate
        for candidate in candidates
    }

    positions = sorted(candidate_by_x)

    cut_combinations = []

    for cut_count in range(1, config.max_cuts_per_component + 1):
        cut_combinations.extend(
            combinations(positions, cut_count)
        )

    for cuts in cut_combinations:
        parts = _make_parts(
            component,
            cuts,
            config.min_part_width,
            config.min_part_area,
        )

        if parts is None:
            continue

        score = float(
            sum(
                candidate_by_x[position].valley_depth
                for position in cuts
            )
        )

        strategy = (
            "single_cut"
            if len(cuts) == 1
            else "multiple_cut"
        )

        hypothesis = SegmentationHypothesis(
            hypothesis_id=(
                f"c{component_id}_"
                + "_".join(str(position) for position in cuts)
            ),
            component_id=component_id,
            strategy=strategy,
            cut_positions=tuple(cuts),
            parts=parts,
            geometry_score=score,
            is_original=False,
        )

        hypotheses.append(hypothesis)

        if len(hypotheses) >= config.max_hypotheses_per_component:
            break

    return ComponentHypothesisSet(
        component_id=component_id,
        original_bbox=original_bbox,
        hypotheses=hypotheses,
        image_bbox=image_bbox,
    )


def generate_image_hypotheses(
    binary: np.ndarray,
    config: HypothesisConfig = HypothesisConfig(),
    segmentation_config: V4SegmentationConfig = DEFAULT_CONFIG,
) -> list[ComponentHypothesisSet]:
    """Generate independent hypotheses for each connected component."""
    binary = validate_binary_image(binary)

    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    components = []

    for label in range(1, count):
        x, y, width, height, area = (
            int(value) for value in stats[label]
        )

        if area < segmentation_config.min_component_area:
            continue

        crop = np.where(
            labels[y:y + height, x:x + width] == label,
            255,
            0,
        ).astype(np.uint8)

        components.append(
            (
                x,
                y,
                label,
                crop,
            )
        )

    components.sort(key=lambda item: (item[0], item[1]))

    results = []

    for x, y, label, crop in components:
        height, width = crop.shape
        results.append(
            generate_component_hypotheses(
                crop,
                component_id=label,
                config=config,
                segmentation_config=segmentation_config,
                image_bbox=(x, y, width, height),
            )
        )

    return results


def hypotheses_to_dict(
    result: ComponentHypothesisSet,
) -> dict[str, Any]:
    """Serialize metadata without embedding image arrays."""
    return {
        "component_id": result.component_id,
        "original_bbox": result.original_bbox,
        "image_bbox": result.image_bbox,
        "hypotheses": [
            {
                "hypothesis_id": item.hypothesis_id,
                "component_id": item.component_id,
                "strategy": item.strategy,
                "cut_positions": item.cut_positions,
                "geometry_score": item.geometry_score,
                "is_original": item.is_original,
                "automatically_accepted": item.automatically_accepted,
                "parts": [
                    {
                        "part_index": part.part_index,
                        "bbox": part.bbox,
                        "area": part.area,
                    }
                    for part in item.parts
                ],
            }
            for item in result.hypotheses
        ],
    }
