from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from scripts.evaluate_v2_v3_benchmark import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    load_manifest,
)
from src.preprocess_v3 import preprocess_v3
from src.segmentation_v3 import segment_v3
from src.v2_inference import V2Predictor


OUTPUT_DIR = Path(
    "artifacts/experiments/cnn_robustness_v3/uncertainty_audit"
)

GEOMETRY_ASPECT_THRESHOLD = 0.80


def main() -> None:
    rows = load_manifest(DEFAULT_MANIFEST)

    predictor = V2Predictor(
        checkpoint_path=str(DEFAULT_CHECKPOINT)
    )

    component_records = []
    sample_records = []

    component_categories = Counter()
    sample_categories = Counter()

    provisional_exact_matches = 0
    total_candidate_splits = 0

    print("=" * 70)
    print("NUMERIQ V3 COMPONENT-LEVEL UNCERTAINTY AUDIT")
    print("=" * 70)
    print("Samples:", len(rows))
    print("Device:", predictor.device)

    for sample_index, row in enumerate(rows, start=1):
        image_path = Path(row["resolved_image_path"])
        ground_truth = row["ground_truth"]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        processed = preprocess_v3(image)
        segmentation = segment_v3(processed["binary"])
        components = segmentation["components"]

        crops = [component["image"] for component in components]
        predictions = []

        for start in range(0, len(crops), 128):
            predictions.extend(
                predictor.predict_crops(crops[start:start + 128])
            )

        if len(predictions) != len(components):
            raise RuntimeError(
                f"Prediction count mismatch for {row['image_path']}"
            )

        provisional_sequence = "".join(
            str(prediction["digit"])
            for prediction in predictions
        )

        sample_has_cnn = False
        sample_has_geometry = False

        for position, (component, prediction) in enumerate(
            zip(components, predictions)
        ):
            aspect_ratio = float(component["aspect_ratio"])
            has_split_candidates = bool(
                component["split_candidates"]
            )

            cnn_uncertain = bool(prediction["is_uncertain"])

            geometry_uncertain = (
                has_split_candidates
                and aspect_ratio >= GEOMETRY_ASPECT_THRESHOLD
            )

            if cnn_uncertain and geometry_uncertain:
                category = "both"
            elif cnn_uncertain:
                category = "cnn_only"
            elif geometry_uncertain:
                category = "geometry_only"
            else:
                category = "neither"

            component_categories[category] += 1
            sample_has_cnn |= cnn_uncertain
            sample_has_geometry |= geometry_uncertain

            split_count = len(component["split_candidates"])
            total_candidate_splits += split_count

            component_records.append({
                "image_path": row["image_path"],
                "ground_truth": ground_truth,
                "component_position": position,
                "predicted_digit": str(prediction["digit"]),
                "confidence": float(prediction["confidence"]),
                "cnn_uncertain": cnn_uncertain,
                "geometry_uncertain": geometry_uncertain,
                "uncertainty_category": category,
                "aspect_ratio": aspect_ratio,
                "has_split_candidates": has_split_candidates,
                "candidate_split_count": split_count,
                "bbox": json.dumps(
                    [int(value) for value in component["bbox"]]
                ),
                "area": int(component["area"]),
                "top3": json.dumps(prediction["top3"]),
            })

        if sample_has_cnn and sample_has_geometry:
            sample_category = "both"
        elif sample_has_cnn:
            sample_category = "cnn_only"
        elif sample_has_geometry:
            sample_category = "geometry_only"
        else:
            sample_category = "neither"

        sample_categories[sample_category] += 1

        provisional_match = provisional_sequence == ground_truth
        provisional_exact_matches += int(provisional_match)

        sample_records.append({
            "image_path": row["image_path"],
            "ground_truth": ground_truth,
            "provisional_sequence": provisional_sequence,
            "provisional_exact_match": provisional_match,
            "component_count": len(components),
            "cnn_uncertain": sample_has_cnn,
            "geometry_uncertain": sample_has_geometry,
            "uncertainty_category": sample_category,
            "candidate_split_count": segmentation[
                "candidate_split_count"
            ],
        })

        if sample_index % 100 == 0 or sample_index == len(rows):
            print(f"Processed {sample_index}/{len(rows)} images")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    component_csv = OUTPUT_DIR / "component_uncertainty.csv"
    sample_csv = OUTPUT_DIR / "sample_uncertainty.csv"
    report_path = OUTPUT_DIR / "uncertainty_audit.json"

    for output_path, records in (
        (component_csv, component_records),
        (sample_csv, sample_records),
    ):
        if records:
            with output_path.open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(records[0].keys()),
                )
                writer.writeheader()
                writer.writerows(records)

    total_components = len(component_records)
    total_samples = len(sample_records)

    report = {
        "evaluation_type": "v3_component_uncertainty_diagnostic",
        "sample_count": total_samples,
        "component_count": total_components,
        "cnn_confidence_threshold": predictor.confidence_threshold,
        "geometry_aspect_ratio_threshold": GEOMETRY_ASPECT_THRESHOLD,
        "component_categories": dict(component_categories),
        "component_category_rates": {
            key: value / total_components
            for key, value in component_categories.items()
        } if total_components else {},
        "sample_categories": dict(sample_categories),
        "sample_category_rates": {
            key: value / total_samples
            for key, value in sample_categories.items()
        } if total_samples else {},
        "provisional_sequence_exact_matches": provisional_exact_matches,
        "provisional_sequence_exact_accuracy": (
            provisional_exact_matches / total_samples
            if total_samples else None
        ),
        "total_candidate_splits": total_candidate_splits,
        "limitations": [
            "This is a diagnostic on composed MNIST sequences, not natural handwriting.",
            "Component-level predictions are not guaranteed to correspond to individual ground-truth digits.",
            "Provisional sequence accuracy does not establish that the segmentation is correct.",
            "Softmax confidence is not a calibrated probability of correctness.",
            "No model, checkpoint, threshold, or recognizer behavior was changed.",
        ],
        "files": {
            "component_csv": str(component_csv),
            "sample_csv": str(sample_csv),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("COMPONENT UNCERTAINTY")
    print("=" * 70)

    for category in (
        "cnn_only",
        "geometry_only",
        "both",
        "neither",
    ):
        print(
            f"{category}:",
            component_categories[category],
        )

    print("\nSAMPLE UNCERTAINTY")

    for category in (
        "cnn_only",
        "geometry_only",
        "both",
        "neither",
    ):
        print(
            f"{category}:",
            sample_categories[category],
        )

    print(
        "\nProvisional exact accuracy:",
        f"{report['provisional_sequence_exact_accuracy']:.2%}",
    )
    print("Total candidate splits:", total_candidate_splits)
    print("Component report:", component_csv)
    print("Sample report:", sample_csv)
    print("JSON report:", report_path)
    print("\nDiagnostic only. No recognition behavior was changed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
