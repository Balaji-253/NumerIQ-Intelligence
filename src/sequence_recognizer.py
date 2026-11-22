from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from src.inference import Predictor
from src.vision import segment_digits


class SequenceRecognizer:
    """
    Recognize sequences of separated handwritten digits.

    Uses:
      - OpenCV connected-component segmentation
      - Existing trained DigitCNN through Predictor
      - Existing image preprocessing and inference pipeline

    Limitation:
      Connected or touching digits are not reliably segmented.
    """

    def __init__(self, predictor: Predictor | None = None):
        self.predictor = predictor or Predictor()

    def recognize(self, image: Image.Image) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image.")

        segments = segment_digits(image)

        if not segments:
            return {
                "status": "no_digits_detected",
                "sequence": "",
                "digit_count": 0,
                "digits": [],
                "warnings": [
                    "No recognizable foreground components were detected."
                ],
            }

        predictions = []

        for index, segment in enumerate(segments):
            crop = segment["image"]

            if (
                not isinstance(crop, np.ndarray)
                or crop.size == 0
            ):
                continue

            crop_image = Image.fromarray(crop).convert("L")

            prediction = self.predictor.predict(crop_image)

            predictions.append({
                "position": index,
                "digit": str(prediction["digit"]),
                "confidence": float(prediction["confidence"]),
                "is_uncertain": bool(
                    prediction.get("is_uncertain", False)
                ),
                "bbox": segment["bbox"],
                "area": segment["area"],
                "top3": prediction.get("top3", []),
            })

        if not predictions:
            return {
                "status": "recognition_failed",
                "sequence": "",
                "digit_count": 0,
                "digits": [],
                "warnings": ["No components could be classified."],
            }

        sequence = "".join(
            item["digit"] for item in predictions
        )

        uncertain_positions = [
            item["position"]
            for item in predictions
            if item["is_uncertain"]
        ]

        warnings = [
            "Connected or overlapping digits may be merged "
            "or incorrectly segmented.",
            "Recognition accuracy on real handwriting has not "
            "yet been independently validated.",
        ]

        if uncertain_positions:
            warnings.append(
                "Low-confidence predictions at positions: "
                + ", ".join(map(str, uncertain_positions))
            )

        return {
            "status": (
                "completed_with_uncertainty"
                if uncertain_positions
                else "completed"
            ),
            "sequence": sequence,
            "digit_count": len(predictions),
            "digits": predictions,
            "warnings": warnings,
        }
