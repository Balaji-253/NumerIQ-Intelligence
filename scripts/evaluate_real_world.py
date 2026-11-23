from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence_evaluation import summarize_predictions
from src.v2_inference import V2Predictor
from src.v2_sequence_recognizer import V2SequenceRecognizer


DEFAULT_CHECKPOINT = Path(
    "artifacts/experiments/cnn_robustness_v2/training/best_model.pt"
)


def load_real_world_manifest(path: Path) -> list[dict[str, str]]:
    """Load labeled samples and restrict image paths to the dataset folder."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    dataset_root = path.parent.resolve()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {"image_path", "ground_truth"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Manifest requires image_path and ground_truth columns."
            )

        rows = []
        seen_paths = set()

        for line_number, row in enumerate(reader, start=2):
            image_path = (row.get("image_path") or "").strip()
            truth = (row.get("ground_truth") or "").strip()

            if not image_path:
                raise ValueError(f"Missing image_path at line {line_number}.")

            if not truth or not truth.isdigit():
                raise ValueError(
                    f"ground_truth must contain digits only at line {line_number}."
                )

            resolved = (dataset_root / image_path).resolve()

            if not resolved.is_relative_to(dataset_root):
                raise ValueError(
                    f"Image path escapes dataset directory at line {line_number}."
                )

            if not resolved.is_file():
                raise FileNotFoundError(
                    f"Image not found at line {line_number}: {resolved}"
                )

            key = str(resolved).casefold()
            if key in seen_paths:
                raise ValueError(f"Duplicate image path at line {line_number}.")
            seen_paths.add(key)

            rows.append({
                "image_path": image_path,
                "ground_truth": truth,
                "writer_id": (row.get("writer_id") or "").strip() or "unknown",
                "capture_method": (row.get("capture_method") or "").strip(),
                "writing_instrument": (
                    row.get("writing_instrument") or ""
                ).strip(),
                "paper_type": (row.get("paper_type") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
                "_resolved_path": str(resolved),
            })

    return rows


def evaluate(
    manifest_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    print("\nNUMERIQ V2 — REAL-WORLD EVALUATION")
    print("=" * 48)

    rows = load_real_world_manifest(manifest_path)

    if not rows:
        result = {
            "status": "WAITING_FOR_REAL_HANDWRITING_SAMPLES",
            "total_samples": 0,
            "message": (
                "No real handwriting samples are available. "
                "No model was loaded and no accuracy was calculated."
            ),
        }
        print(json.dumps(result, indent=2))
        return result

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"V2 checkpoint not found: {checkpoint_path}"
        )

    print(f"Samples: {len(rows)}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Loading V2 model on the available inference device...")

    predictor = V2Predictor(checkpoint_path=str(checkpoint_path))
    recognizer = V2SequenceRecognizer(predictor)

    predictions: list[dict[str, Any]] = []

    for index, sample in enumerate(rows, start=1):
        started = time.perf_counter()

        try:
            with Image.open(sample["_resolved_path"]) as source:
                image = source.convert("RGB").copy()

            result = recognizer.recognize(image)
            latency_ms = (time.perf_counter() - started) * 1000

            confidences = [
                float(digit["confidence"])
                for digit in result["digits"]
            ]

            prediction = {
                "image_path": sample["image_path"],
                "writer_id": sample["writer_id"],
                "capture_method": sample["capture_method"],
                "writing_instrument": sample["writing_instrument"],
                "paper_type": sample["paper_type"],
                "ground_truth": sample["ground_truth"],
                "prediction": result["sequence"],
                "status": result["status"],
                "digit_count": result["digit_count"],
                "mean_confidence": (
                    statistics.fmean(confidences) if confidences else None
                ),
                "low_confidence": any(
                    digit["is_uncertain"] for digit in result["digits"]
                ),
                "latency_ms": round(latency_ms, 3),
                "warnings": " | ".join(result.get("warnings", [])),
                "error": "",
            }

        except Exception as error:
            latency_ms = (time.perf_counter() - started) * 1000

            prediction = {
                "image_path": sample["image_path"],
                "writer_id": sample["writer_id"],
                "capture_method": sample["capture_method"],
                "writing_instrument": sample["writing_instrument"],
                "paper_type": sample["paper_type"],
                "ground_truth": sample["ground_truth"],
                "prediction": "",
                "status": "processing_error",
                "digit_count": 0,
                "mean_confidence": None,
                "low_confidence": False,
                "latency_ms": round(latency_ms, 3),
                "warnings": "",
                "error": f"{type(error).__name__}: {error}",
            }

        predictions.append(prediction)

        print(
            f"[{index}/{len(rows)}] "
            f"truth={sample['ground_truth']} "
            f"prediction={prediction['prediction'] or '<empty>'} "
            f"status={prediction['status']}"
        )

    metrics = summarize_predictions(predictions)

    writer_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        writer_groups[prediction["writer_id"]].append(prediction)

    per_writer = {
        writer_id: summarize_predictions(writer_predictions)
        for writer_id, writer_predictions in sorted(writer_groups.items())
    }

    metrics.update({
        "status": "EVALUATION_COMPLETED",
        "evaluation_type": "independent_real_world_handwriting",
        "model_metadata": predictor.model_metadata,
        "checkpoint": str(checkpoint_path),
        "dataset": {
            "manifest": str(manifest_path),
            "total_samples": len(rows),
            "distinct_writers": len(writer_groups),
            "capture_methods": sorted({
                row["capture_method"] for row in rows
                if row["capture_method"]
            }),
            "writing_instruments": sorted({
                row["writing_instrument"] for row in rows
                if row["writing_instrument"]
            }),
            "paper_types": sorted({
                row["paper_type"] for row in rows if row["paper_type"]
            }),
        },
        "per_writer": per_writer,
        "interpretation": (
            "Metrics apply only to the independently collected samples "
            "listed in this manifest. They must not be interpreted as "
            "general population accuracy. These samples are evaluation-only "
            "and are not used for training or fine-tuning."
        ),
    })

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "real_world_v2_metrics.json"
    predictions_path = output_dir / "real_world_v2_predictions.csv"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with predictions_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(predictions[0].keys()),
        )
        writer.writeheader()
        writer.writerows(predictions)

    print("\nOVERALL REAL-WORLD RESULTS")
    print("=" * 48)
    print(json.dumps(metrics, indent=2))
    print(f"\nSummary: {summary_path}")
    print(f"Predictions: {predictions_path}")

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate NumerIQ V2 on independent real handwriting."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/real_world_handwriting/manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/real_world_handwriting/reports"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    args = parser.parse_args()

    try:
        evaluate(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
        )
        return 0
    except Exception as error:
        print(
            f"EVALUATION_ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
