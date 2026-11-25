from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from src.v2_inference import V2Predictor
from src.v2_sequence_recognizer import V2SequenceRecognizer
from src.v3_sequence_recognizer import V3SequenceRecognizer


PROJECT_ROOT = Path.cwd()

DEFAULT_MANIFEST = Path("data/handwritten_sequences/manifest.csv")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/experiments/cnn_robustness_v3/benchmark_comparison"
)
DEFAULT_CHECKPOINT = (
    "artifacts/experiments/cnn_robustness_v2/training/best_model.pt"
)


def levenshtein_distance(reference: str, prediction: str) -> int:
    previous = list(range(len(prediction) + 1))

    for i, ref_char in enumerate(reference, start=1):
        current = [i]

        for j, pred_char in enumerate(prediction, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_char != pred_char),
                )
            )

        previous = current

    return previous[-1]


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    index = max(0, int((percent / 100) * len(ordered) + 0.999999) - 1)
    return ordered[index]


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"image_path", "ground_truth"}

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Manifest must contain image_path and ground_truth columns."
            )

        for line_number, row in enumerate(reader, start=2):
            image_path = (row.get("image_path") or "").strip()
            ground_truth = (row.get("ground_truth") or "").strip()

            if not image_path:
                raise ValueError(
                    f"Missing image_path at manifest line {line_number}."
                )

            if not ground_truth.isdigit():
                raise ValueError(
                    f"Invalid ground_truth at manifest line {line_number}."
                )

            resolved_path = (path.resolve().parent / image_path).resolve()

            if not resolved_path.is_relative_to(PROJECT_ROOT.resolve()):
                raise ValueError(
                    f"Image path escapes project directory: {image_path}"
                )

            if not resolved_path.is_file():
                raise FileNotFoundError(
                    f"Benchmark image not found: {resolved_path}"
                )

            rows.append({
                "image_path": image_path,
                "resolved_image_path": str(resolved_path),
                "ground_truth": ground_truth,
            })

    if not rows:
        raise ValueError("Benchmark manifest contains no samples.")

    return rows


def run_recognizer(recognizer: Any, image: Image.Image) -> dict[str, Any]:
    start = time.perf_counter()

    try:
        result = recognizer.recognize(image)
        latency_ms = (time.perf_counter() - start) * 1000

        status = result.get("status", "unknown")
        failed = status == "recognition_failed"

        return {
            "prediction": result.get("sequence", "") if not failed else "",
            "status": status,
            "digit_count": int(result.get("digit_count", 0)),
            "uncertain": (
                status == "completed_with_uncertainty"
                or any(
                    bool(item.get("is_uncertain", False))
                    for item in result.get("digits", [])
                )
            ),
            "processing_failure": failed,
            "latency_ms": round(latency_ms, 3),
            "warnings": result.get("warnings", []),
        }

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "prediction": "",
            "status": "processing_error",
            "digit_count": 0,
            "uncertain": False,
            "processing_failure": True,
            "latency_ms": round(latency_ms, 3),
            "warnings": [f"{type(exc).__name__}: {exc}"],
        }


def calculate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)

    exact_matches = sum(
        item["prediction"] == item["ground_truth"]
        for item in records
    )

    edit_distance = sum(
        levenshtein_distance(
            item["ground_truth"],
            item["prediction"],
        )
        for item in records
    )

    reference_characters = sum(
        len(item["ground_truth"]) for item in records
    )

    similarities = [
        1 - (
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
        for item in records
    ]

    count_mismatches = sum(
        item["digit_count"] != len(item["ground_truth"])
        for item in records
    )

    failures = sum(item["processing_failure"] for item in records)
    uncertain = sum(item["uncertain"] for item in records)

    nonempty = [
        item for item in records
        if item["ground_truth"]
    ]

    no_detection = sum(
        item["digit_count"] == 0 for item in nonempty
    )

    latencies = [
        item["latency_ms"]
        for item in records
        if item["latency_ms"] is not None
    ]

    return {
        "total_samples": total,
        "exact_matches": exact_matches,
        "exact_sequence_accuracy": exact_matches / total,
        "total_character_edit_distance": edit_distance,
        "total_reference_characters": reference_characters,
        "character_error_rate": (
            edit_distance / reference_characters
            if reference_characters else None
        ),
        "normalized_edit_similarity": statistics.fmean(similarities),
        "segmentation_unit_count_matches": total - count_mismatches,
        "segmentation_unit_count_agreement": (
            1 - count_mismatches / total
        ),
        "processing_failures": failures,
        "processing_failure_rate": failures / total,
        "uncertain_samples": uncertain,
        "uncertain_sample_rate": uncertain / total,
        "no_detection_on_nonempty_samples": no_detection,
        "no_detection_rate_on_nonempty_samples": (
            no_detection / len(nonempty) if nonempty else None
        ),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)

    rows = load_manifest(manifest_path)

    print("=" * 72)
    print("NUMERIQ — V2 vs V3 COMPOSED SEQUENCE BENCHMARK")
    print("=" * 72)
    print("Samples:", len(rows))
    print("Manifest:", manifest_path)
    print("Checkpoint:", checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"V2 checkpoint not found: {checkpoint_path}"
        )

    print("\nLoading shared V2 CNN checkpoint...")

    predictor = V2Predictor(checkpoint_path=str(checkpoint_path))
    v2_recognizer = V2SequenceRecognizer(predictor)
    v3_recognizer = V3SequenceRecognizer(predictor)

    print("Model:", predictor.model_metadata["model_name"])
    print("Device:", predictor.device)

    v2_records = []
    v3_records = []
    per_image = []

    for index, row in enumerate(rows, start=1):
        image_path = Path(row["resolved_image_path"])
        ground_truth = row["ground_truth"]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        v2 = run_recognizer(v2_recognizer, image)
        v3 = run_recognizer(v3_recognizer, image)

        for engine, result, collection in (
            ("v2", v2, v2_records),
            ("v3", v3, v3_records),
        ):
            collection.append({
                "ground_truth": ground_truth,
                **result,
            })

        record = {
            "image_path": row["image_path"],
            "ground_truth": ground_truth,
            "sequence_length": len(ground_truth),
            "v2_prediction": v2["prediction"],
            "v2_exact_match": v2["prediction"] == ground_truth,
            "v2_status": v2["status"],
            "v2_segmentation_unit_count": v2["digit_count"],
            "v2_count_agreement": v2["digit_count"] == len(ground_truth),
            "v2_uncertain": v2["uncertain"],
            "v2_processing_failure": v2["processing_failure"],
            "v2_latency_ms": v2["latency_ms"],
            "v2_warnings": v2["warnings"],
            "v3_prediction": v3["prediction"],
            "v3_exact_match": v3["prediction"] == ground_truth,
            "v3_status": v3["status"],
            "v3_segmentation_unit_count": v3["digit_count"],
            "v3_count_agreement": v3["digit_count"] == len(ground_truth),
            "v3_uncertain": v3["uncertain"],
            "v3_processing_failure": v3["processing_failure"],
            "v3_latency_ms": v3["latency_ms"],
            "v3_warnings": v3["warnings"],
        }

        per_image.append(record)

        if index % 100 == 0 or index == len(rows):
            print(f"Processed {index}/{len(rows)} images")

    v2_metrics = calculate_metrics(v2_records)
    v3_metrics = calculate_metrics(v3_records)

    by_length: dict[str, Any] = {}

    grouped: dict[int, list[int]] = defaultdict(list)

    for index, record in enumerate(per_image):
        grouped[record["sequence_length"]].append(index)

    for length, indices in sorted(grouped.items()):
        by_length[str(length)] = {
            "sample_count": len(indices),
            "v2": calculate_metrics([v2_records[i] for i in indices]),
            "v3": calculate_metrics([v3_records[i] for i in indices]),
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "v2_v3_predictions.csv"
    json_path = output_dir / "v2_v3_comparison.json"

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(per_image[0].keys()),
        )
        writer.writeheader()
        writer.writerows(per_image)

    summary = {
        "benchmark_name": "NumerIQ MNIST Composed Sequence Benchmark",
        "evaluation_type": "composed_mnist_sequence_benchmark",
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "model": predictor.model_metadata,
        "device": str(predictor.device),
        "sample_count": len(rows),
        "v2": v2_metrics,
        "v3": v3_metrics,
        "by_sequence_length": by_length,
        "interpretation": {
            "exact_sequence_accuracy": (
                "Every sample is included. Uncertain or empty predictions "
                "on nonempty ground truth count as incorrect."
            ),
            "segmentation_unit_count_agreement": (
                "V2 counts grouped segments; V3 counts connected components. "
                "This is not verified handwritten digit-count accuracy."
            ),
            "uncertainty": (
                "V3 uncertainty includes possible merged-component ambiguity; "
                "V2 uncertainty is based on classifier uncertainty."
            ),
            "latency": (
                "Measured locally on this machine and includes recognizer "
                "inference, but excludes model loading and image file opening."
            ),
        },
        "limitations": [
            "The images are composed from isolated MNIST official test digits.",
            "The benchmark is not naturally written multi-digit handwriting.",
            "It does not establish writer-independent real-world OCR accuracy.",
            "Repeated source digit images may occur across sequences.",
            "Results apply to this benchmark and this checkpoint only.",
            "The benchmark must not be used to tune thresholds and then "
            "reported as an independent final test.",
        ],
        "files": {
            "predictions_csv": str(csv_path),
        },
    }

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS")
    print("=" * 72)

    for name, metrics in (("V2", v2_metrics), ("V3", v3_metrics)):
        print(f"\n{name}")
        print(
            "Exact sequence accuracy:",
            f"{metrics['exact_sequence_accuracy']:.2%}",
        )
        print("Character error rate:", f"{metrics['character_error_rate']:.4f}")
        print(
            "Segmentation-unit count agreement:",
            f"{metrics['segmentation_unit_count_agreement']:.2%}",
        )
        print("Uncertain sample rate:", f"{metrics['uncertain_sample_rate']:.2%}")
        print("Processing failures:", metrics["processing_failures"])
        print("Mean latency:", f"{metrics['latency_ms']['mean']:.2f} ms")
        print("P95 latency:", f"{metrics['latency_ms']['p95']:.2f} ms")

    print("\nJSON report:", json_path)
    print("Per-image CSV:", csv_path)
    print("\nComposed benchmark only — not real-world handwriting validation.")
    print("=" * 72)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare V2 and V3 on the composed MNIST sequence benchmark."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
    )

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
