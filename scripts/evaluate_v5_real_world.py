from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_real_world import load_real_world_manifest
from src.v4_pipeline import V4Pipeline
from src.v5_pipeline import V5Pipeline


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))

    for i, ref_char in enumerate(reference, start=1):
        current = [i]

        for j, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_char != hyp_char),
            ))

        previous = current

    return previous[-1]


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    index = max(0, math.ceil(percent * len(ordered)) - 1)
    return ordered[index]


def evaluate(
    manifest_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
    beam_width: int = 64,
    max_results: int = 20,
) -> dict[str, Any]:
    print("\nNUMERIQ V5 — REAL-HANDWRITING EVALUATION")
    print("=" * 52)

    rows = load_real_world_manifest(manifest_path)

    if not rows:
        raise ValueError("The evaluation manifest contains no samples.")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"V2 checkpoint not found: {checkpoint_path}"
        )

    print(f"Samples: {len(rows)}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Loading V5 pipeline and existing V2 CNN...")

    v4_pipeline = V4Pipeline(checkpoint_path=str(checkpoint_path))
    pipeline = V5Pipeline(
        v4_pipeline=v4_pipeline,
        beam_width=beam_width,
        max_results=max_results,
    )

    predictions: list[dict[str, Any]] = []

    for index, sample in enumerate(rows, start=1):
        started = time.perf_counter()

        try:
            with Image.open(sample["_resolved_path"]) as source:
                image = source.convert("RGB").copy()

            result = pipeline.analyze(image)
            decoding = result["sequence_decoding"]
            candidates = decoding.get("candidates", [])

            truth = sample["ground_truth"]
            candidate_strings = [
                item["sequence"] for item in candidates
            ]

            top1 = candidate_strings[0] if candidate_strings else ""
            matching_rank = next(
                (
                    rank
                    for rank, sequence in enumerate(
                        candidate_strings, start=1
                    )
                    if sequence == truth
                ),
                None,
            )

            latency_ms = (time.perf_counter() - started) * 1000

            predictions.append({
                "image_path": sample["image_path"],
                "writer_id": sample["writer_id"],
                "ground_truth": truth,
                "top1_prediction": top1,
                "top1_exact": top1 == truth,
                "correct_candidate_rank": matching_rank,
                "hit_at_5": matching_rank is not None and matching_rank <= 5,
                "hit_at_10": matching_rank is not None and matching_rank <= 10,
                "hit_at_20": matching_rank is not None and matching_rank <= 20,
                "top1_edit_distance": edit_distance(truth, top1),
                "top1_cer": edit_distance(truth, top1) / len(truth),
                "top1_digit_count": len(top1),
                "true_digit_count": len(truth),
                "digit_count_correct": len(top1) == len(truth),
                "candidate_count": len(candidates),
                "candidate_sequences": json.dumps(candidate_strings),
                "ordering_ambiguity": decoding.get(
                    "ordering_ambiguity", False
                ),
                "latency_ms": round(latency_ms, 3),
                "status": result["status"],
                "error": "",
            })

        except Exception as error:
            latency_ms = (time.perf_counter() - started) * 1000

            predictions.append({
                "image_path": sample["image_path"],
                "writer_id": sample["writer_id"],
                "ground_truth": sample["ground_truth"],
                "top1_prediction": "",
                "top1_exact": False,
                "correct_candidate_rank": None,
                "hit_at_5": False,
                "hit_at_10": False,
                "hit_at_20": False,
                "top1_edit_distance": len(sample["ground_truth"]),
                "top1_cer": 1.0,
                "top1_digit_count": 0,
                "true_digit_count": len(sample["ground_truth"]),
                "digit_count_correct": False,
                "candidate_count": 0,
                "candidate_sequences": "[]",
                "ordering_ambiguity": False,
                "latency_ms": round(latency_ms, 3),
                "status": "processing_error",
                "error": f"{type(error).__name__}: {error}",
            })

        prediction = predictions[-1]

        print(
            f"[{index}/{len(rows)}] "
            f"truth={sample['ground_truth']} "
            f"top1={prediction['top1_prediction'] or '<empty>'} "
            f"correct_rank={prediction['correct_candidate_rank']} "
            f"status={prediction['status']}"
        )

    total = len(predictions)
    successful = [
        item for item in predictions
        if item["status"] != "processing_error"
    ]
    latencies = [item["latency_ms"] for item in predictions]

    total_reference_characters = sum(
        len(item["ground_truth"]) for item in predictions
    )
    total_edit_distance = sum(
        item["top1_edit_distance"] for item in predictions
    )

    metrics = {
        "status": "EVALUATION_COMPLETED",
        "evaluation_type": "real_world_handwriting_v5_experimental",
        "total_samples": total,
        "successfully_evaluated": len(successful),
        "processing_failures": total - len(successful),
        "top1_exact_sequence_accuracy": (
            sum(item["top1_exact"] for item in predictions) / total
        ),
        "candidate_recall_at_5": (
            sum(item["hit_at_5"] for item in predictions) / total
        ),
        "candidate_recall_at_10": (
            sum(item["hit_at_10"] for item in predictions) / total
        ),
        "candidate_recall_at_20": (
            sum(item["hit_at_20"] for item in predictions) / total
        ),
        "character_error_rate": (
            total_edit_distance / total_reference_characters
            if total_reference_characters else 0.0
        ),
        "total_character_edit_distance": total_edit_distance,
        "total_reference_characters": total_reference_characters,
        "digit_count_accuracy": (
            sum(item["digit_count_correct"] for item in predictions) / total
        ),
        "mean_latency_ms": statistics.fmean(latencies),
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "mean_candidate_count": statistics.fmean(
            item["candidate_count"] for item in predictions
        ),
        "ordering_ambiguity_rate": (
            sum(item["ordering_ambiguity"] for item in successful)
            / len(successful)
            if successful else 0.0
        ),
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "beam_width": beam_width,
        "max_results": max_results,
        "dataset": {
            "total_samples": total,
            "distinct_writers": len({
                row["writer_id"] for row in rows
            }),
            "capture_methods": sorted({
                row["capture_method"] for row in rows
                if row["capture_method"]
            }),
        },
        "interpretation": (
            "Results apply only to the samples in this manifest. "
            "V5 is experimental and reuses the V2 CNN. Candidate recall "
            "measures whether the correct sequence appears in the returned "
            "candidate list; it does not establish recognition reliability. "
            "These evaluation samples must not be used for training or tuning."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "v5_real_handwriting_metrics.json"
    predictions_path = output_dir / "v5_real_handwriting_predictions.csv"

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

    print("\nV5 REAL-HANDWRITING RESULTS")
    print("=" * 52)
    print(json.dumps(metrics, indent=2))
    print(f"\nSummary: {summary_path}")
    print(f"Predictions: {predictions_path}")

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate experimental NumerIQ V5 on real handwriting."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/v4_handwriting_challenge/v2_evaluation_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/experiments/cnn_robustness_v4/evaluation/"
            "v5_sequence_decoder"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/experiments/cnn_robustness_v2/training/best_model.pt"
        ),
    )
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--max-results", type=int, default=20)

    args = parser.parse_args()

    try:
        evaluate(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            beam_width=args.beam_width,
            max_results=args.max_results,
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
