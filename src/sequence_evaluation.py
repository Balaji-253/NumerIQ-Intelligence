from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from PIL import Image

from src.inference import Predictor
from src.sequence_recognizer import SequenceRecognizer


def levenshtein_distance(reference: str, prediction: str) -> int:
    """Calculate character-level Levenshtein edit distance."""
    previous = list(range(len(prediction) + 1))

    for i, ref_char in enumerate(reference, start=1):
        current = [i]

        for j, pred_char in enumerate(prediction, start=1):
            substitution = previous[j - 1] + (
                ref_char != pred_char
            )
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1

            current.append(
                min(substitution, deletion, insertion)
            )

        previous = current

    return previous[-1]


def percentile(
    values: list[float],
    percent: float,
) -> float | None:
    """Calculate a nearest-rank percentile."""
    if not 0 <= percent <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    if not values:
        return None

    ordered = sorted(values)
    index = max(
        0,
        math.ceil((percent / 100) * len(ordered)) - 1,
    )

    return ordered[index]


def load_manifest(path: Path) -> list[dict[str, str]]:
    """Load and validate the labeled image manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"image_path", "ground_truth"}

        if not reader.fieldnames or not required.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                "Manifest must contain image_path and ground_truth columns."
            )

        rows = []

        for line_number, row in enumerate(reader, start=2):
            image_path = (row.get("image_path") or "").strip()
            ground_truth = (row.get("ground_truth") or "").strip()

            if not image_path:
                raise ValueError(
                    f"Missing image_path at line {line_number}."
                )

            if not ground_truth or not ground_truth.isdigit():
                raise ValueError(
                    f"ground_truth must contain digits only at "
                    f"line {line_number}."
                )

            rows.append({
                "image_path": image_path,
                "ground_truth": ground_truth,
            })

    if not rows:
        raise ValueError("Manifest contains no evaluation samples.")

    return rows


def summarize_predictions(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate failure-aware evaluation metrics.

    Every submitted sample contributes to exact sequence accuracy
    and character error rate. A processing failure is an empty
    prediction, not an excluded sample.
    """
    if not predictions:
        raise ValueError("Cannot evaluate an empty prediction collection.")

    total = len(predictions)

    failures = [
        item for item in predictions
        if item["status"] == "processing_error"
    ]

    successful = [
        item for item in predictions
        if item["status"] != "processing_error"
    ]

    exact_matches = sum(
        item["prediction"] == item["ground_truth"]
        for item in predictions
    )

    total_edit_distance = sum(
        levenshtein_distance(
            item["ground_truth"],
            item["prediction"],
        )
        for item in predictions
    )

    reference_characters = sum(
        len(item["ground_truth"])
        for item in predictions
    )

    similarities = [
        1.0 - (
            levenshtein_distance(
                item["ground_truth"],
                item["prediction"],
            )
            / max(
                len(item["ground_truth"]),
                len(item["prediction"]),
                1,
            )
        )
        for item in predictions
    ]

    count_mismatches = sum(
        item["digit_count"] != len(item["ground_truth"])
        for item in successful
    )

    count_mismatches_all = sum(
        item["digit_count"] != len(item["ground_truth"])
        for item in predictions
    )

    no_detections = sum(
        item["digit_count"] == 0
        for item in successful
    )

    low_confidence = sum(
        bool(item["low_confidence"])
        for item in successful
    )

    latencies = [
        float(item["latency_ms"])
        for item in predictions
        if item["latency_ms"] is not None
    ]

    return {
        "total_samples": total,
        "successfully_evaluated": len(successful),
        "processing_failures": len(failures),
        "processing_failure_rate": len(failures) / total,
        "exact_sequence_accuracy": exact_matches / total,
        "character_error_rate": (
            total_edit_distance / reference_characters
            if reference_characters else None
        ),
        "normalized_edit_similarity": statistics.fmean(similarities),
        "segmentation_count_mismatch_rate": (
            count_mismatches / len(successful)
            if successful else None
        ),
        "digit_count_mismatch_rate_all": (
            count_mismatches_all / total
        ),
        "digit_count_mismatch_rate_successful": (
            count_mismatches / len(successful)
            if successful else None
        ),
        "digit_count_accuracy_all": (
            1.0 - count_mismatches_all / total
        ),
        "no_detection_rate": (
            no_detections / len(successful)
            if successful else None
        ),
        "low_confidence_sample_rate": (
            low_confidence / len(successful)
            if successful else None
        ),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "total_character_edit_distance": total_edit_distance,
        "total_reference_characters": reference_characters,
    }


def evaluate(
    manifest_path: Path,
    output_dir: Path,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Evaluate the complete image-to-sequence inference pipeline."""
    rows = load_manifest(manifest_path)

    predictor = Predictor(checkpoint_path=checkpoint_path)
    recognizer = SequenceRecognizer(predictor)

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = []

    for sample in rows:
        relative_path = Path(sample["image_path"])

        image_path = (
            relative_path
            if relative_path.is_absolute()
            else manifest_path.parent / relative_path
        )

        truth = sample["ground_truth"]
        started = time.perf_counter()

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB").copy()

            result = recognizer.recognize(image)
            latency_ms = (time.perf_counter() - started) * 1000

            confidences = [
                digit["confidence"]
                for digit in result["digits"]
            ]

            predictions.append({
                "image_path": sample["image_path"],
                "ground_truth": truth,
                "prediction": result["sequence"],
                "status": result["status"],
                "digit_count": result["digit_count"],
                "mean_confidence": (
                    statistics.fmean(confidences)
                    if confidences else None
                ),
                "low_confidence": any(
                    digit["is_uncertain"]
                    for digit in result["digits"]
                ),
                "latency_ms": round(latency_ms, 3),
                "error": "",
            })

        except Exception as error:
            latency_ms = (time.perf_counter() - started) * 1000

            logger_message = (
                f"{type(error).__name__}: {error}"
            )

            predictions.append({
                "image_path": sample["image_path"],
                "ground_truth": truth,
                "prediction": "",
                "status": "processing_error",
                "digit_count": 0,
                "mean_confidence": None,
                "low_confidence": False,
                "latency_ms": round(latency_ms, 3),
                "error": logger_message,
            })

    metrics = summarize_predictions(predictions)

    metrics.update({
        "evaluation_type": "handwritten_sequence_recognition",
        "checkpoint": checkpoint_path,
        "model_metadata": predictor.model_metadata,
        "interpretation": (
            "Results apply only to this labeled dataset and inference "
            "pipeline. They do not establish general real-world accuracy."
        ),
    })

    summary_path = output_dir / "sequence_evaluation.json"
    predictions_path = output_dir / "sequence_predictions.csv"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    fieldnames = list(predictions[0].keys())

    with predictions_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(json.dumps(metrics, indent=2))
    print(f"\nSummary: {summary_path}")
    print(f"Predictions: {predictions_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate handwritten multi-digit recognition."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
    )

    parser.add_argument(
        "--checkpoint",
        default="artifacts/mnist_cnn.pt",
    )

    args = parser.parse_args()

    evaluate(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
