from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import datasets
from torch.utils.data import random_split

from src.vision import BOUNDING_BOX_PADDING, enhance_image
from experiments.grouped_vision_v2 import (
    UnionFind,
    component_gaps,
    extract_components_v2,
    HORIZONTAL_GAP_THRESHOLD,
    VERTICAL_GAP_THRESHOLD,
)


SEED = 42
COMPOSITION_SEED = 2026

TRAIN_SIZE = 55000
VALIDATION_SIZE = 5000

SEQUENCE_LENGTH = 4
DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28
CANVAS_HEIGHT = 64
DIGIT_GAP = 10

IMAGE_SIZE = 28
DIGIT_SIZE = 20

MEAN = 0.1307
STD = 0.3081

OUTPUT_DIR = Path(
    "artifacts/experiments/cnn_robustness_v1/crops"
)


def normalize_crop(crop: np.ndarray) -> np.ndarray:
    """Apply the exact direct mass-centered preprocessing."""

    if crop.ndim != 2 or crop.size == 0:
        raise ValueError("Invalid crop dimensions.")

    foreground = crop > 0

    if not foreground.any():
        raise ValueError("Crop has no foreground.")

    ys, xs = np.where(foreground)

    cropped = crop[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1,
    ]

    image = Image.fromarray(cropped.astype(np.uint8), mode="L")

    image.thumbnail(
        (DIGIT_SIZE, DIGIT_SIZE),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)

    left = (IMAGE_SIZE - image.width) // 2
    top = (IMAGE_SIZE - image.height) // 2

    canvas.paste(image, (left, top))

    array = np.asarray(canvas, dtype=np.float32).copy()
    total_mass = float(array.sum())

    if total_mass > 0:
        ys, xs = np.indices(array.shape)

        center_x = float((xs * array).sum() / total_mass)
        center_y = float((ys * array).sum() / total_mass)

        shift_x = int(round(13.5 - center_x))
        shift_y = int(round(13.5 - center_y))

        canvas = canvas.transform(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Transform.AFFINE,
            (1, 0, -shift_x, 0, 1, -shift_y),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

    return np.asarray(canvas, dtype=np.uint8).copy()


def segment_with_component_metadata(
    image: Image.Image,
) -> tuple[list[dict], dict[int, dict]]:
    """
    Reproduce V2 grouping using the same enhancement,
    component extraction, grouping thresholds and padding.
    """

    processed = enhance_image(image)
    binary = processed["binary"]

    height, width = binary.shape
    components = extract_components_v2(binary)

    if not components:
        return [], {}

    component_map = {
        item["id"]: item
        for item in components
    }

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

    groups: dict[int, list[dict]] = {}

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
            "image": crop,
            "component_ids": [
                item["id"] for item in group
            ],
        })

    return results, component_map


def compose_sequence(dataset, indices: list[int]):
    """Compose a four-digit sequence and retain its source labels."""

    canvas_width = (
        SEQUENCE_LENGTH * DIGIT_WIDTH
        + (SEQUENCE_LENGTH - 1) * DIGIT_GAP
    )

    canvas = Image.new(
        "L",
        (canvas_width, CANVAS_HEIGHT),
        color=255,
    )

    labels = []

    for position, index in enumerate(indices):
        digit_image, label = dataset[index]
        digit_image = ImageOps.invert(digit_image.convert("L"))

        if digit_image.size != (DIGIT_WIDTH, DIGIT_HEIGHT):
            raise RuntimeError("Unexpected MNIST digit dimensions.")

        x = position * (DIGIT_WIDTH + DIGIT_GAP)
        y = (CANVAS_HEIGHT - DIGIT_HEIGHT) // 2

        canvas.paste(digit_image, (x, y))
        labels.append(int(label))

    return canvas, labels


def associate_segments(
    segments: list[dict],
    component_map: dict[int, dict],
) -> dict[int, np.ndarray] | None:
    """
    Associate exactly one segment with each source digit.

    Components in the 10-pixel inter-digit gaps are ambiguous.
    Multiple segments assigned to one digit are also rejected.
    """

    tile_segments: dict[int, list[np.ndarray]] = {
        index: []
        for index in range(SEQUENCE_LENGTH)
    }

    tile_width = DIGIT_WIDTH + DIGIT_GAP

    for segment in segments:
        component_tiles = set()

        for component_id in segment["component_ids"]:
            component = component_map.get(component_id)

            if component is None:
                return None

            center_x = component["centroid"][0]
            tile_index = int(center_x // tile_width)

            if not 0 <= tile_index < SEQUENCE_LENGTH:
                return None

            tile_left = tile_index * tile_width

            if not (
                tile_left <= center_x < tile_left + DIGIT_WIDTH
            ):
                return None

            component_tiles.add(tile_index)

        if len(component_tiles) != 1:
            return None

        tile_index = next(iter(component_tiles))
        tile_segments[tile_index].append(segment["image"])

    if any(len(items) != 1 for items in tile_segments.values()):
        return None

    return {
        index: items[0]
        for index, items in tile_segments.items()
    }


def prepare_partition(
    dataset,
    source_indices: list[int],
    partition: str,
) -> dict:
    """Generate disk-backed crop arrays incrementally."""

    image_path = OUTPUT_DIR / f"{partition}_images.npy"
    label_path = OUTPUT_DIR / f"{partition}_labels.npy"
    manifest_path = OUTPUT_DIR / f"{partition}_manifest.csv"

    image_store = np.lib.format.open_memmap(
        image_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(source_indices), IMAGE_SIZE, IMAGE_SIZE),
    )

    label_store = np.lib.format.open_memmap(
        label_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(source_indices),),
    )

    rng = random.Random(
        COMPOSITION_SEED + (0 if partition == "train" else 1)
    )

    shuffled_indices = list(source_indices)
    rng.shuffle(shuffled_indices)

    if len(shuffled_indices) % SEQUENCE_LENGTH != 0:
        raise RuntimeError("Partition cannot form complete sequences.")

    accepted = 0
    rejected_sequences = 0
    rejected_digits = 0
    sequence_count = len(shuffled_indices) // SEQUENCE_LENGTH

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sequence_id",
                "source_index",
                "position",
                "label",
            ],
        )
        writer.writeheader()

        for sequence_number in range(sequence_count):
            start = sequence_number * SEQUENCE_LENGTH
            indices = shuffled_indices[
                start:start + SEQUENCE_LENGTH
            ]

            image, labels = compose_sequence(dataset, indices)

            segments, component_map = segment_with_component_metadata(
                image
            )

            associated = associate_segments(
                segments,
                component_map,
            )

            if associated is None:
                rejected_sequences += 1
                rejected_digits += SEQUENCE_LENGTH
                continue

            for position in range(SEQUENCE_LENGTH):
                crop = normalize_crop(associated[position])

                image_store[accepted] = crop
                label_store[accepted] = labels[position]

                writer.writerow({
                    "sequence_id": sequence_number,
                    "source_index": indices[position],
                    "position": position,
                    "label": labels[position],
                })

                accepted += 1

            if (sequence_number + 1) % 250 == 0:
                image_store.flush()
                label_store.flush()

                print(
                    f"{partition}: "
                    f"{sequence_number + 1}/{sequence_count} "
                    f"sequences | accepted crops: {accepted}",
                    flush=True,
                )

    image_store.flush()
    label_store.flush()

    del image_store
    del label_store

    return {
        "partition": partition,
        "source_images": len(source_indices),
        "sequences_attempted": sequence_count,
        "accepted_crops": accepted,
        "rejected_sequences": rejected_sequences,
        "rejected_digits": rejected_digits,
        "image_array": str(image_path),
        "label_array": str(label_path),
        "manifest": str(manifest_path),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv2.setNumThreads(2)
    torch.set_num_threads(2)

    print("Loading MNIST training split...", flush=True)

    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=False,
        transform=None,
    )

    if len(dataset) != 60000:
        raise RuntimeError("Expected exactly 60,000 MNIST training images.")

    train_subset, validation_subset = random_split(
        dataset,
        [TRAIN_SIZE, VALIDATION_SIZE],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_indices = list(train_subset.indices)
    validation_indices = list(validation_subset.indices)

    if set(train_indices).intersection(validation_indices):
        raise RuntimeError("Training/validation data leakage detected.")

    print(f"Training source images: {len(train_indices)}")
    print(f"Validation source images: {len(validation_indices)}")
    print("Generating training crops...", flush=True)

    train_result = prepare_partition(
        dataset,
        train_indices,
        "train",
    )

    print("\nGenerating validation crops...", flush=True)

    validation_result = prepare_partition(
        dataset,
        validation_indices,
        "validation",
    )

    metadata = {
        "experiment": "NumerIQ CNN Robustness V1",
        "seed": SEED,
        "composition_seed": COMPOSITION_SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "digit_gap": DIGIT_GAP,
        "canvas_height": CANVAS_HEIGHT,
        "source_dataset": "MNIST official training split",
        "normalization": "direct mass-centered V2 normalization",
        "segmentation": "V2 enhancement and component grouping",
        "train": train_result,
        "validation": validation_result,
        "test_split_used": False,
        "original_checkpoint_modified": False,
        "limitations": [
            "Generated sequences are composed from isolated MNIST digits.",
            "This dataset does not represent natural multi-digit handwriting.",
            "Ambiguous segmentation associations are rejected.",
            "Real-world OCR performance requires independent validation.",
        ],
    }

    metadata_path = OUTPUT_DIR / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n" + "=" * 65)
    print("NUMERIQ V1 CROP DATASET GENERATION COMPLETE")
    print("=" * 65)
    print(json.dumps(metadata, indent=2))
    print(f"\nMetadata: {metadata_path}")


if __name__ == "__main__":
    main()
