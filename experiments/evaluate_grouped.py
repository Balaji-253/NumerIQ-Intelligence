from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from PIL import Image

from src.inference import Predictor
from src.sequence_evaluation import (
    levenshtein_distance,
    load_manifest,
    percentile,
)
from experiments.grouped_vision import segment_digits_grouped


def evaluate(
    manifest_path: Path,
    output_dir: Path,
    checkpoint_path: str,
) -> dict:
    rows = load_manifest(manifest_path)
    predictor = Predictor(checkpoint_path=checkpoint_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = []

    print(f"Evaluation samples: {len(rows)}")
    print(f"Device: {predictor.device}")
    print("Starting experimental grouped evaluation...")

    for index, sample in enumerate(rows, start=1):
        image_path = manifest_path.parent / sample["image_path"]
        truth = sample["ground_truth"]
        started = time.perf_counter()

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB").copy()

            segments = segment_digits_grouped(image)

            digit_results = []

            for segment in segments:
                crop = Image.fromarray(segment["image"]).convert("L")
                result = predictor.predict(crop)

                digit_results.append({
                    "digit": str(result["digit"]),
                    "confidence": result["confidence"],
                    "top3": result["top3"],
                    "component_count": segment["component_count"],
                    "bbox": segment["bbox"],
                })

            prediction = "".join(
                item["digit"] for item in digit_results
            )

            confidences = [
                item["confidence"] for item in digit_results
            ]

            latency_ms = (time.perf_counter() - started) * 1000

            predictions.append({
                "image": image_path.name,
                "ground_truth": truth,
                "prediction": prediction,
                "exact": prediction == truth,
                "edit_distance": levenshtein_distance(
                    truth, prediction
                ),
                "digit_count": len(digit_results),
                "component_count": sum(
                    item["component_count"] for item in digit_results
                ),
                "mean_confidence": (
                    statistics.fmean(confidences)
                    if confidences else None
                ),
                "low_confidence": any(
                    confidence < predictor.confidence_threshold
                    for confidence in confidences
                ),
                "latency_ms": round(latency_ms, 3),
                "status": "completed",
                "error": "",
                "digit_details": json.dumps(digit_results),
            })

        except Exception as error:
            latency_ms = (time.perf_counter() - started) * 1000

            predictions.append({
                "image": image_path.name,
                "ground_truth": truth,
                "prediction": "",
                "exact": False,
                "edit_distance": len(truth),
                "digit_count": 0,
                "component_count": 0,
                "mean_confidence": None,
                "low_confidence": False,
                "latency_ms": round(latency_ms, 3),
                "status": "processing_error",
                "error": f"{type(error).__name__}: {error}",
                "digit_details": "[]",
            })

        if index % 100 == 0:
            print(f"Processed {index}/{len(rows)}")

    total = len(predictions)
    failures = sum(
        item["status"] == "processing_error"
        for item in predictions
    )

    exact_matches = sum(item["exact"] for item in predictions)

    total_edits = sum(
        item["edit_distance"] for item in predictions
    )

    total_reference_characters = sum(
        len(item["ground_truth"]) for item in predictions
    )

    successful = [
        item for item in predictions
        if item["status"] == "completed"
    ]

    count_mismatches = sum(
        item["digit_count"] != len(item["ground_truth"])
        for item in successful
    )

    latencies = [
        item["latency_ms"] for item in predictions
    ]

    metrics = {
        "evaluation_type": "experimental_grouped_segmentation",
        "total_samples": total,
        "successful_samples": len(successful),
        "processing_failures": failures,
        "exact_sequence_accuracy": exact_matches / total,
        "character_error_rate": (
            total_edits / total_reference_characters
            if total_reference_characters else None
        ),
        "total_character_edit_distance": total_edits,
        "total_reference_characters": total_reference_characters,
        "segmentation_count_mismatch_rate": (
            count_mismatches / len(successful)
            if successful else None
        ),
        "low_confidence_sample_rate": (
            sum(item["low_confidence"] for item in successful)
            / len(successful)
            if successful else None
        ),
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "checkpoint": checkpoint_path,
        "model_metadata": predictor.model_metadata,
        "limitations": [
            "This benchmark uses artificially composed MNIST sequences.",
            "This is an experimental grouping implementation.",
            "Results do not establish real-world handwriting accuracy.",
            "The benchmark must be independently validated.",
        ],
    }

    metrics_path = output_dir / "grouped_evaluation.json"
    predictions_path = output_dir / "grouped_predictions.csv"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    fieldnames = list(predictions[0].keys())

    with predictions_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print("\n" + "=" * 60)
    print("EXPERIMENTAL GROUPED EVALUATION")
    print("=" * 60)
    print(f"Samples: {total}")
    print(f"Failures: {failures}")
    print(f"Exact sequence accuracy: {metrics['exact_sequence_accuracy']:.2%}")
    print(f"Character error rate: {metrics['character_error_rate']:.2%}")
    print(
        "Segmentation count mismatch: "
        f"{metrics['segmentation_count_mismatch_rate']:.2%}"
    )
    print(f"Mean latency: {metrics['latency_ms']['mean']:.3f} ms")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate experimental grouped digit segmentation."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/handwritten_sequences/manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/grouped_reproduction"),
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
