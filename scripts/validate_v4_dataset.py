from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    PROJECT_ROOT
    / "data/v4_handwriting_challenge/manifests/handwriting_manifest.csv"
)

SPEC_PATH = (
    PROJECT_ROOT
    / "artifacts/experiments/cnn_robustness_v4/dataset/dataset_spec.json"
)

MIN_DIMENSION = 8
MAX_DIMENSION = 6000

REQUIRED_FIELDS = {
    "sample_id",
    "image_path",
    "ground_truth",
    "source_type",
    "writer_id",
    "distortion_type",
    "capture_method",
    "split",
    "ground_truth_verified",
}

def validate() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    if not MANIFEST.is_file():
        print(f"ERROR: Manifest not found: {MANIFEST}")
        return 1

    if not SPEC_PATH.is_file():
        print(f"ERROR: Dataset specification not found: {SPEC_PATH}")
        return 1

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    categories = set(spec["distortion_categories"])
    source_types = set(spec["source_types"])
    splits = set(spec["splits"])
    capture_methods = set(spec["capture_methods"])
    supported_formats = set(spec["supported_formats"])

    sample_ids: set[str] = set()
    image_paths: set[str] = set()
    writers_by_split: dict[str, set[str]] = defaultdict(set)

    counts = defaultdict(int)
    verified_count = 0

    with MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            errors.append("Manifest has no header.")
            rows = []
        else:
            missing = REQUIRED_FIELDS - set(reader.fieldnames)

            if missing:
                errors.append(
                    f"Missing required columns: {sorted(missing)}"
                )

            rows = list(reader)

    for row_number, row in enumerate(rows, start=2):
        prefix = f"Row {row_number}"

        sample_id = (row.get("sample_id") or "").strip()
        image_value = (row.get("image_path") or "").strip()
        label = (row.get("ground_truth") or "").strip()
        source_type = (row.get("source_type") or "").strip().lower()
        writer_id = (row.get("writer_id") or "").strip()
        distortion = (row.get("distortion_type") or "").strip().lower()
        capture = (row.get("capture_method") or "").strip().lower()
        split = (row.get("split") or "").strip().lower()
        verified = (
            (row.get("ground_truth_verified") or "")
            .strip()
            .lower()
        )

        if not sample_id:
            errors.append(f"{prefix}: Missing sample_id.")
        elif sample_id in sample_ids:
            errors.append(f"{prefix}: Duplicate sample_id: {sample_id}")
        else:
            sample_ids.add(sample_id)

        if not image_value:
            errors.append(f"{prefix}: Missing image_path.")
            continue

        relative_path = Path(image_value)

        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(
                f"{prefix}: Unsafe image path: {image_value}"
            )
            continue

        normalized_path = relative_path.as_posix().lower()

        if normalized_path in image_paths:
            errors.append(
                f"{prefix}: Duplicate image_path: {image_value}"
            )
        else:
            image_paths.add(normalized_path)

        image_path = PROJECT_ROOT / relative_path

        if image_path.suffix.lower() not in supported_formats:
            errors.append(
                f"{prefix}: Unsupported image format: {image_path.suffix}"
            )
            continue

        if not image_path.is_file():
            errors.append(
                f"{prefix}: Image file does not exist: {image_value}"
            )
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                width, height = image.size

            if not (
                MIN_DIMENSION <= width <= MAX_DIMENSION
                and MIN_DIMENSION <= height <= MAX_DIMENSION
            ):
                errors.append(
                    f"{prefix}: Invalid image dimensions: "
                    f"{width}x{height}"
                )

        except Exception as exc:
            errors.append(
                f"{prefix}: Invalid image: {image_value} ({exc})"
            )

        if not label or not label.isdigit():
            errors.append(
                f"{prefix}: Ground truth must contain digits 0-9 only."
            )

        if source_type not in source_types:
            errors.append(
                f"{prefix}: Invalid source_type: {source_type}"
            )

        if distortion not in categories:
            errors.append(
                f"{prefix}: Invalid distortion_type: {distortion}"
            )

        if capture not in capture_methods:
            errors.append(
                f"{prefix}: Invalid capture_method: {capture}"
            )

        if split not in splits:
            errors.append(
                f"{prefix}: Invalid split: {split}"
            )

        if source_type == "real" and not writer_id:
            errors.append(
                f"{prefix}: Real handwriting requires an anonymous writer_id."
            )

        if (
            source_type == "synthetic"
            and writer_id
        ):
            warnings.append(
                f"{prefix}: Synthetic sample has a writer_id; "
                "check whether this is appropriate."
            )

        if (
            source_type == "real"
            and writer_id
            and split in splits
        ):
            writers_by_split[split].add(writer_id)

        if verified == "true":
            verified_count += 1
        elif verified == "false":
            warnings.append(
                f"{prefix}: Ground truth is not independently verified."
            )
        else:
            errors.append(
                f"{prefix}: ground_truth_verified must be true or false."
            )

        if source_type in source_types and split in splits:
            counts[(source_type, split)] += 1

    split_names = list(splits)

    for index, first in enumerate(split_names):
        for second in split_names[index + 1:]:
            overlap = writers_by_split[first] & writers_by_split[second]

            if overlap:
                errors.append(
                    f"Writer leakage between {first} and {second}: "
                    f"{len(overlap)} writer IDs overlap."
                )

    print("=" * 65)
    print("NUMERIQ V4 DATASET VALIDATION")
    print("=" * 65)
    print("Manifest:", MANIFEST)
    print("Total samples:", len(rows))
    print("Unique sample IDs:", len(sample_ids))
    print("Verified annotations:", verified_count)

    print("\nDATA DISTRIBUTION")

    for source_type in sorted(source_types):
        for split in sorted(splits):
            print(
                f"{source_type:10} | {split:10} | "
                f"{counts[(source_type, split)]}"
            )

    print("\nREAL WRITER COUNTS")

    for split in sorted(splits):
        print(
            f"{split:10}: {len(writers_by_split[split])} writers"
        )

    print("\nWARNINGS:", len(warnings))

    for warning in warnings[:20]:
        print("WARNING:", warning)

    if len(warnings) > 20:
        print("Additional warnings:", len(warnings) - 20)

    print("\nERRORS:", len(errors))

    for error in errors[:30]:
        print("ERROR:", error)

    if len(errors) > 30:
        print("Additional errors:", len(errors) - 30)

    if errors:
        print("\nRESULT: VALIDATION FAILED")
        return 1

    if not rows:
        print(
            "\nRESULT: STRUCTURE VALID — NO SAMPLES AVAILABLE YET."
        )
        return 0

    print("\nRESULT: DATASET STRUCTURE VALID")
    print("Note: Structural validation does not prove annotation correctness.")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
