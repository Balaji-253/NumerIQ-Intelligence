from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.v2_inference import V2Predictor
from src.v3_sequence_recognizer import V3SequenceRecognizer


OUTPUT_DIR = Path(
    "artifacts/experiments/cnn_robustness_v3"
)

IMAGE_DIR = OUTPUT_DIR / "diagnostic_images"


def load_font(size: int = 64):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/times.ttf",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue

    return ImageFont.load_default()


def create_base_image(width=260, height=140):
    return Image.new("RGB", (width, height), "white")


def draw_digit(draw, digit, position, font):
    draw.text(
        position,
        digit,
        font=font,
        fill="black",
    )


def create_test_images():
    font = load_font(64)
    cases = {}

    # 1. Single digit
    image = create_base_image(180, 120)
    draw_digit(ImageDraw.Draw(image), "8", (55, 15), font)

    cases["single_digit"] = {
        "image": image,
        "ground_truth": "8",
        "category": "single",
    }

    # 2. Separated digits
    image = create_base_image()
    draw = ImageDraw.Draw(image)

    draw_digit(draw, "1", (30, 20), font)
    draw_digit(draw, "8", (145, 20), font)

    cases["separated_digits"] = {
        "image": image,
        "ground_truth": "18",
        "category": "separated",
    }

    # 3. Closely spaced digits
    image = create_base_image(190, 120)
    draw = ImageDraw.Draw(image)

    draw_digit(draw, "1", (30, 20), font)
    draw_digit(draw, "8", (78, 20), font)

    cases["close_digits"] = {
        "image": image,
        "ground_truth": "18",
        "category": "close",
    }

    # 4. Rotated digits
    image = create_base_image(220, 150)
    draw = ImageDraw.Draw(image)

    draw_digit(draw, "1", (35, 35), font)
    draw_digit(draw, "8", (105, 35), font)

    image = image.rotate(
        12,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor="white",
    )

    cases["rotated_digits"] = {
        "image": image,
        "ground_truth": "18",
        "category": "rotated",
    }

    # 5. Slanted digits using horizontal shear
    image = create_base_image(240, 150)
    draw = ImageDraw.Draw(image)

    draw_digit(draw, "1", (35, 25), font)
    draw_digit(draw, "8", (115, 25), font)

    image = image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, -0.20, 25, 0, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor="white",
    )

    cases["slanted_digits"] = {
        "image": image,
        "ground_truth": "18",
        "category": "slanted",
    }

    # 6. Diagonal arrangement
    image = create_base_image(220, 180)
    draw = ImageDraw.Draw(image)

    draw_digit(draw, "1", (25, 15), font)
    draw_digit(draw, "8", (100, 85), font)

    cases["diagonal_digits"] = {
        "image": image,
        "ground_truth": "18",
        "category": "diagonal",
    }

    # 7. Touching/overlapping glyphs
    image = create_base_image(170, 120)
    draw = ImageDraw.Draw(image)

    draw_digit(draw, "1", (35, 20), font)
    draw_digit(draw, "8", (52, 20), font)

    cases["touching_digits"] = {
        "image": image,
        "ground_truth": "18",
        "category": "touching",
    }

    # 8. Blank image
    image = create_base_image(200, 100)

    cases["blank_image"] = {
        "image": image,
        "ground_truth": "",
        "category": "blank",
    }

    return cases


def main():
    print("=" * 70)
    print("NUMERIQ V3 - CONTROLLED DIAGNOSTIC EVALUATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    cases = create_test_images()

    print("\nSaving diagnostic images...")

    for name, case in cases.items():
        case["image"].save(IMAGE_DIR / f"{name}.png")

    print("Saved", len(cases), "images.")

    print("\nLoading existing V2 CNN checkpoint...")

    predictor = V2Predictor()
    recognizer = V3SequenceRecognizer(predictor)

    print("Model:", predictor.model_metadata["model_name"])
    print("Device:", predictor.device)

    results = {}
    exact_matches = 0
    count_matches = 0
    uncertain_cases = 0

    for name, case in cases.items():
        image = case["image"]
        ground_truth = case["ground_truth"]

        start_time = time.perf_counter()
        result = recognizer.recognize(image)
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        predicted = result["sequence"]

        exact_match = predicted == ground_truth
        count_match = (
            result["digit_count"] == len(ground_truth)
        )

        exact_matches += int(exact_match)
        count_matches += int(count_match)

        is_uncertain = (
            result["status"] == "completed_with_uncertainty"
        )

        uncertain_cases += int(is_uncertain)

        case_result = {
            "ground_truth": ground_truth,
            "prediction": predicted,
            "exact_match": exact_match,
            "ground_truth_digit_count": len(ground_truth),
            "predicted_digit_count": result["digit_count"],
            "digit_count_match": count_match,
            "digit_count_is_reliable": (
                result["digit_count_is_reliable"]
            ),
            "status": result["status"],
            "component_count": result.get("component_count", 0),
            "candidate_split_count": (
                result.get("candidate_split_count", 0)
            ),
            "latency_ms": round(latency_ms, 2),
            "digits": result["digits"],
            "warnings": result["warnings"],
        }

        results[name] = case_result

        print("\n" + "-" * 55)
        print("TEST:", name)
        print("Ground truth:", repr(ground_truth))
        print("Prediction:", repr(predicted))
        print("Exact match:", exact_match)
        print("Digit count match:", count_match)
        print("Status:", result["status"])
        print("Components:", result.get("component_count", 0))
        print(
            "Candidate splits:",
            result.get("candidate_split_count", 0),
        )
        print("Latency:", round(latency_ms, 2), "ms")

    total = len(cases)

    summary = {
        "total_cases": total,
        "exact_matches": exact_matches,
        "exact_sequence_accuracy": exact_matches / total,
        "digit_count_matches": count_matches,
        "digit_count_accuracy": count_matches / total,
        "uncertain_cases": uncertain_cases,
        "model": predictor.model_metadata,
        "evaluation_type": "controlled_synthetic_diagnostic",
        "limitations": [
            "Synthetic printed-font images are not natural handwriting.",
            "Eight cases are insufficient for statistical accuracy claims.",
            "The test does not establish writer-independent performance.",
            "Touching and overlapping cases require visual inspection.",
        ],
        "cases": results,
    }

    report_path = OUTPUT_DIR / "real_model_diagnostic.json"

    report_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print("Total cases:", total)
    print("Exact matches:", exact_matches)
    print(
        "Exact sequence accuracy:",
        f"{summary['exact_sequence_accuracy']:.2%}",
    )
    print(
        "Digit-count accuracy:",
        f"{summary['digit_count_accuracy']:.2%}",
    )
    print("Uncertain cases:", uncertain_cases)
    print("Report:", report_path)
    print("Images:", IMAGE_DIR)
    print("\nSynthetic diagnostic only - not production validation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
