from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.vision import BOUNDING_BOX_PADDING, enhance_image


HORIZONTAL_GAP_THRESHOLD = 6
VERTICAL_GAP_THRESHOLD = 8


class UnionFind:
    """Disjoint-set structure for experimental component association."""

    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: int, second: int) -> None:
        root_a = self.find(first)
        root_b = self.find(second)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def extract_components_v2(
    binary: np.ndarray,
) -> list[dict[str, Any]]:
    """Extract all non-background components without early noise filtering."""

    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    components = []

    for index in range(1, count):
        x, y, w, h, area = stats[index]

        if area <= 0:
            continue

        components.append({
            "id": int(index),
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": int(area),
            "centroid": [
                float(centroids[index][0]),
                float(centroids[index][1]),
            ],
        })

    return components


def component_gaps(
    first: dict[str, Any],
    second: dict[str, Any],
) -> tuple[int, int]:
    """Calculate horizontal and vertical bounding-box separation."""

    x1, y1, w1, h1 = first["bbox"]
    x2, y2, w2, h2 = second["bbox"]

    horizontal_gap = max(
        0,
        max(x1, x2) - min(x1 + w1, x2 + w2),
    )

    vertical_gap = max(
        0,
        max(y1, y2) - min(y1 + h1, y2 + h2),
    )

    return horizontal_gap, vertical_gap


def segment_digits_grouped_v2(
    image: Image.Image,
) -> list[dict[str, Any]]:
    """
    Experimental geometry-based grouping.

    Retains small components and associates components when both
    bounding-box gaps satisfy the configured thresholds.

    This implementation is experimental and not production validated.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    processed = enhance_image(image)
    binary = processed["binary"]

    height, width = binary.shape
    components = extract_components_v2(binary)

    if not components:
        return []

    union_find = UnionFind(len(components))

    for first_index in range(len(components)):
        for second_index in range(first_index + 1, len(components)):
            horizontal_gap, vertical_gap = component_gaps(
                components[first_index],
                components[second_index],
            )

            if (
                horizontal_gap <= HORIZONTAL_GAP_THRESHOLD
                and vertical_gap <= VERTICAL_GAP_THRESHOLD
            ):
                union_find.union(first_index, second_index)

    groups: dict[int, list[dict[str, Any]]] = {}

    for index, component in enumerate(components):
        root = union_find.find(index)
        groups.setdefault(root, []).append(component)

    results = []

    for group in groups.values():
        left = min(item["bbox"][0] for item in group)
        top = min(item["bbox"][1] for item in group)

        right = max(
            item["bbox"][0] + item["bbox"][2]
            for item in group
        )

        bottom = max(
            item["bbox"][1] + item["bbox"][3]
            for item in group
        )

        padded_left = max(0, left - BOUNDING_BOX_PADDING)
        padded_top = max(0, top - BOUNDING_BOX_PADDING)
        padded_right = min(width, right + BOUNDING_BOX_PADDING)
        padded_bottom = min(height, bottom + BOUNDING_BOX_PADDING)

        crop = binary[
            padded_top:padded_bottom,
            padded_left:padded_right,
        ].copy()

        if crop.size == 0:
            continue

        results.append({
            "bbox": [
                int(padded_left),
                int(padded_top),
                int(padded_right - padded_left),
                int(padded_bottom - padded_top),
            ],
            "image": crop,
            "area": int(sum(item["area"] for item in group)),
            "component_count": len(group),
            "component_ids": [item["id"] for item in group],
        })

    results.sort(
        key=lambda item: (
            item["bbox"][0],
            item["bbox"][1],
        )
    )

    return results
