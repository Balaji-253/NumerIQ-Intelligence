from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from torchvision import datasets


SEED = 2026
SAMPLES_PER_LENGTH = 200
SEQUENCE_LENGTHS = (2, 3, 4, 5, 6)
DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28
CANVAS_HEIGHT = 64
DIGIT_GAP = 10

OUTPUT_DIR = Path("data/handwritten_sequences")
IMAGES_DIR = OUTPUT_DIR / "images"


def main() -> None:
    rng = random.Random(SEED)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    dataset = datasets.MNIST(
        root="data",
        train=False,
        download=False,
    )

    if len(dataset) != 10000:
        raise RuntimeError(
            f"Expected 10,000 MNIST test images, found {len(dataset)}."
        )

    manifest_path = OUTPUT_DIR / "manifest.csv"
    rows = []
    sequence_counter = 0
    length_counts = Counter()

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "ground_truth"],
        )
        writer.writeheader()

        for sequence_length in SEQUENCE_LENGTHS:
            for _ in range(SAMPLES_PER_LENGTH):
                selected_indices = [
                    rng.randrange(len(dataset))
                    for _ in range(sequence_length)
                ]

                digits = []
                labels = []

                for index in selected_indices:
                    digit_image, label = dataset[index]

                    digit_image = digit_image.convert("L")

                    # MNIST uses white digits on a black background.
                    # Convert to black digits on a white page.
                    digit_image = ImageOps.invert(digit_image)

                    if digit_image.size != (
                        DIGIT_WIDTH,
                        DIGIT_HEIGHT,
                    ):
                        raise RuntimeError(
                            f"Unexpected digit dimensions: "
                            f"{digit_image.size}"
                        )

                    digits.append(digit_image)
                    labels.append(str(int(label)))

                ground_truth = "".join(labels)

                canvas_width = (
                    sequence_length * DIGIT_WIDTH
                    + (sequence_length - 1) * DIGIT_GAP
                )

                canvas = Image.new(
                    "L",
                    (canvas_width, CANVAS_HEIGHT),
                    color=255,
                )

                for position, digit_image in enumerate(digits):
                    x = position * (DIGIT_WIDTH + DIGIT_GAP)
                    y = (CANVAS_HEIGHT - DIGIT_HEIGHT) // 2

                    canvas.paste(digit_image, (x, y))

                sequence_counter += 1

                filename = f"sequence_{sequence_counter:05d}.png"
                image_path = IMAGES_DIR / filename

                canvas.save(image_path, format="PNG")

                relative_path = f"images/{filename}"

                row = {
                    "image_path": relative_path,
                    "ground_truth": ground_truth,
                }

                writer.writerow(row)
                rows.append(row)
                length_counts[sequence_length] += 1

    metadata = {
        "dataset_name": "NumerIQ MNIST Composed Sequence Benchmark",
        "source_dataset": "MNIST official test split",
        "source_test_images": len(dataset),
        "seed": SEED,
        "samples_per_sequence_length": SAMPLES_PER_LENGTH,
        "sequence_lengths": list(SEQUENCE_LENGTHS),
        "total_sequences": sequence_counter,
        "counts_by_length": {
            str(key): value
            for key, value in sorted(length_counts.items())
        },
        "digit_gap_pixels": DIGIT_GAP,
        "canvas_height": CANVAS_HEIGHT,
        "image_polarity": "black digits on white background",
        "limitations": [
            "Images are composed from isolated MNIST test digits.",
            "This is not a naturally handwritten multi-digit dataset.",
            "The benchmark does not establish general real-world OCR accuracy.",
            "Repeated source digit images may occur across sequences.",
        ],
    }

    metadata_path = OUTPUT_DIR / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("=" * 60)
    print("NUMERIQ SEQUENCE BENCHMARK CREATED")
    print("=" * 60)
    print(f"Source images: {len(dataset)}")
    print(f"Generated sequences: {sequence_counter}")

    for length, count in sorted(length_counts.items()):
        print(f"{length}-digit sequences: {count}")

    print(f"Manifest: {manifest_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Images: {IMAGES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
