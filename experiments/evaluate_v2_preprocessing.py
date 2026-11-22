from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.inference import Predictor
from src.sequence_evaluation import (
    levenshtein_distance,
    load_manifest,
    percentile,
)
from experiments.grouped_vision_v2 import segment_digits_grouped_v2


IMAGE_SIZE = 28
DIGIT_SIZE = 20
MEAN = 0.1307
STD = 0.3081


def direct_normalize(
    crop: np.ndarray,
    center_mass: bool,
) -> torch.Tensor:
    """Normalize an existing white-foreground binary crop."""

    if crop.ndim != 2:
        raise ValueError("Expected a two-dimensional binary crop.")

    if crop.size == 0:
        raise ValueError("Empty crop received.")

    foreground = crop > 0

    if not foreground.any():
        raise ValueError("Crop contains no foreground pixels.")

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

    if center_mass:
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

            array = np.asarray(canvas, dtype=np.float32).copy()

    array /= 255.0
    array = (array - MEAN) / STD

    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)

    return tensor


def predict_direct(
    predictor: Predictor,
    crop: np.ndarray,
    center_mass: bool,
) -> dict:
    tensor = direct_normalize(crop, center_mass).to(predictor.device)

    with torch.inference_mode():
        logits = predictor.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        values, indices = torch.topk(probabilities, k=5)

    return {
        "digit": int(indices[0].item()),
        "confidence": float(values[0].item()),
        "top3": [
            {
                "digit": int(index.item()),
                "probability": float(value.item()),
            }
            for value, index in zip(values[:3], indices[:3])
        ],
    }


def evaluate(
    manifest_path: Path,
    output_dir: Path,
    checkpoint_path: str,
) -> dict:
    rows = load_manifest(manifest_path)
    predictor = Predictor(checkpoint_path=checkpoint_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "baseline": None,
        "direct_geometric": False,
        "direct_mass_centered": True,
    }

    predictions = {name: [] for name in methods}

    print(f"Samples: {len(rows)}")
    print(f"Device: {predictor.device}")
    print("Methods:", ", ".join(methods))
    print("Starting controlled preprocessing experiment...")

    for index, sample in enumerate(rows, start=1):
        image_path = manifest_path.parent / sample["image_path"]
        truth = sample["ground_truth"]

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB").copy()

            segments = segment_digits_grouped_v2(image)

            for method, center_mass in methods.items():
                started = time.perf_counter()

                digit_results = []

                for segment in segments:
                    crop = segment["image"]

                    if method == "baseline":
                        pil_crop = Image.fromarray(crop).convert("L")
                        result = predictor.predict(pil_crop)
                    else:
                        result = predict_direct(
                            predictor,
                            crop,
                            center_mass,
                        )

                    digit_results.append(str(result["digit"]))

                prediction = "".join(digit_results)
                elapsed_ms = (time.perf_counter() - started) * 1000

                predictions[method].append({
                    "image": image_path.name,
                    "ground_truth": truth,
                    "prediction": prediction,
                    "exact": prediction == truth,
                    "edit_distance": levenshtein_distance(
                        truth,
                        prediction,
                    ),
                    "digit_count": len(digit_results),
                    "reference_digit_count": len(truth),
                    "count_mismatch": len(digit_results) != len(truth),
                    "latency_ms": round(elapsed_ms, 3),
                    "status": "completed",
                    "error": "",
                })

        except Exception as error:
            for method in methods:
                predictions[method].append({
                    "image": image_path.name,
                    "ground_truth": truth,
                    "prediction": "",
                    "exact": False,
                    "edit_distance": len(truth),
                    "digit_count": 0,
                    "reference_digit_count": len(truth),
                    "count_mismatch": True,
                    "latency_ms": None,
                    "status": "processing_error",
                    "error": f"{type(error).__name__}: {error}",
                })

        if index % 100 == 0:
            print(f"Processed {index}/{len(rows)}")

    total = len(rows)
    results = {}

    for method, records in predictions.items():
        successful = [
            item for item in records
            if item["status"] == "completed"
        ]

        failures = total - len(successful)

        exact_matches = sum(item["exact"] for item in records)
        total_edits = sum(item["edit_distance"] for item in records)
        total_characters = sum(
            len(item["ground_truth"]) for item in records
        )

        latencies = [
            item["latency_ms"]
            for item in successful
            if item["latency_ms"] is not None
        ]

        results[method] = {
            "total_samples": total,
            "successful_samples": len(successful),
            "processing_failures": failures,
            "exact_sequence_accuracy": exact_matches / total,
            "character_error_rate": (
                total_edits / total_characters
                if total_characters else None
            ),
            "segmentation_count_mismatch_rate": (
                sum(item["count_mismatch"] for item in successful)
                / len(successful)
                if successful else None
            ),
            "latency_ms": {
                "mean": statistics.fmean(latencies) if latencies else None,
                "p50": percentile(latencies, 50) if latencies else None,
                "p95": percentile(latencies, 95) if latencies else None,
            },
        }

        csv_path = output_dir / f"{method}_predictions.csv"

        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(records[0].keys()),
            )
            writer.writeheader()
            writer.writerows(records)

    report = {
        "experiment": "v2_preprocessing_comparison",
        "checkpoint": checkpoint_path,
        "device": str(predictor.device),
        "segmentation": {
            "algorithm": "Grouped Vision V2",
            "horizontal_gap_threshold": 6,
            "vertical_gap_threshold": 8,
        },
        "methods": results,
        "limitations": [
            "Benchmark consists of artificially composed MNIST sequences.",
            "All methods use identical V2 segmentation.",
            "This experiment does not establish real-world handwriting accuracy.",
            "Direct normalization methods are experimental.",
        ],
    }

    report_path = output_dir / "preprocessing_comparison.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 65)
    print("V2 PREPROCESSING ACCURACY COMPARISON")
    print("=" * 65)

    for method, metrics in results.items():
        print(f"\n{method}")
        print(
            "Exact sequence accuracy: "
            f"{metrics['exact_sequence_accuracy']:.2%}"
        )
        print(
            "Character error rate: "
            f"{metrics['character_error_rate']:.2%}"
        )
        print(
            "Count mismatch: "
            f"{metrics['segmentation_count_mismatch_rate']:.2%}"
        )
        print(f"Failures: {metrics['processing_failures']}")

    print(f"\nReport: {report_path}")
    print(f"Output directory: {output_dir}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/handwritten_sequences/manifest.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/experiments/v2_preprocessing_comparison"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default="artifacts/mnist_cnn.pt",
    )

    args = parser.parse_args()

    evaluate(
        args.manifest,
        args.output_dir,
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
