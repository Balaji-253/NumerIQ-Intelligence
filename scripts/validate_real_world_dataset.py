from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


ROOT = Path("data/real_world_handwriting")
MANIFEST = ROOT / "manifest.csv"
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}
REQUIRED_FIELDS = {
    "image_path",
    "ground_truth",
    "writer_id",
    "capture_method",
    "writing_instrument",
    "paper_type",
}


def main() -> int:
    if not MANIFEST.is_file():
        print("ERROR: Dataset manifest not found.")
        return 1

    errors = []
    valid_rows = []
    seen_paths = set()

    with MANIFEST.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or not REQUIRED_FIELDS.issubset(
            set(reader.fieldnames)
        ):
            print("ERROR: Manifest schema is invalid.")
            return 1

        rows = list(reader)

    for line, row in enumerate(rows, start=2):
        relative_path = (row.get("image_path") or "").strip()
        label = (row.get("ground_truth") or "").strip()
        writer_id = (row.get("writer_id") or "").strip()

        if not relative_path:
            errors.append(f"Line {line}: missing image_path.")
            continue

        if relative_path in seen_paths:
            errors.append(f"Line {line}: duplicate image_path.")
            continue

        seen_paths.add(relative_path)

        if not label or not label.isdigit():
            errors.append(f"Line {line}: invalid ground_truth.")
            continue

        if not writer_id:
            errors.append(f"Line {line}: missing writer_id.")
            continue

        for field in (
            "capture_method",
            "writing_instrument",
            "paper_type",
        ):
            if not (row.get(field) or "").strip():
                errors.append(f"Line {line}: missing {field}.")

        image_path = (ROOT / relative_path).resolve()

        if not image_path.is_relative_to(ROOT.resolve()):
            errors.append(f"Line {line}: image path escapes dataset.")
            continue

        if not image_path.is_file():
            errors.append(f"Line {line}: image not found: {relative_path}")
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                if image.format not in ALLOWED_FORMATS:
                    errors.append(
                        f"Line {line}: unsupported format {image.format}."
                    )
                    continue

                if image.width < 16 or image.height < 16:
                    errors.append(f"Line {line}: image dimensions too small.")

                if image.width > 6000 or image.height > 6000:
                    errors.append(f"Line {line}: image dimensions too large.")

        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
        ) as error:
            errors.append(f"Line {line}: invalid image ({error}).")
            continue

        valid_rows.append(row)

    print("\nREAL-WORLD DATASET VALIDATION")
    print("=" * 42)
    print("Manifest records:", len(rows))
    print("Valid images:", len(valid_rows))
    print("Distinct writers:", len({
        row["writer_id"].strip() for row in valid_rows
    }))
    print("Errors:", len(errors))

    if valid_rows:
        print("Writer distribution:")
        for writer, count in sorted(
            Counter(row["writer_id"] for row in valid_rows).items()
        ):
            print(f"  {writer}: {count} images")

    for error in errors:
        print("ERROR:", error)

    if not rows:
        print("\nSTATUS: WAITING_FOR_REAL_HANDWRITING_SAMPLES")
        return 0

    if errors:
        print("\nSTATUS: VALIDATION_FAILED")
        return 1

    print("\nSTATUS: DATASET_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
