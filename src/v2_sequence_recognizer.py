from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from experiments.grouped_vision_v2 import segment_digits_grouped_v2
from src.v2_inference import V2Predictor


class V2SequenceRecognizer:
    """Sequence recognition using grouped V2 segmentation and V2 inference."""

    def __init__(self, predictor: V2Predictor):
        self.predictor = predictor

    def recognize(self, image: Image.Image) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image.")

        segments = segment_digits_grouped_v2(image)

        if not segments:
            return {
                "status": "no_digits_detected",
                "sequence": "",
                "digit_count": 0,
                "digits": [],
                "warnings": [
                    "No foreground components were detected."
                ],
            }

        valid_segments = [
            segment
            for segment in segments
            if isinstance(segment.get("image"), np.ndarray)
            and segment["image"].ndim == 2
            and segment["image"].size > 0
        ]

        if not valid_segments:
            return {
                "status": "recognition_failed",
                "sequence": "",
                "digit_count": 0,
                "digits": [],
                "warnings": [
                    "No valid digit crops could be classified."
                ],
            }

        crops = [
            segment["image"]
            for segment in valid_segments
        ]

        predictions = self.predictor.predict_crops(crops)

        digits = []

        for position, (segment, prediction) in enumerate(
            zip(valid_segments, predictions)
        ):
            digits.append({
                "position": position,
                "digit": str(prediction["digit"]),
                "confidence": float(prediction["confidence"]),
                "is_uncertain": bool(
                    prediction["is_uncertain"]
                ),
                "bbox": segment["bbox"],
                "area": int(segment["area"]),
                "component_count": int(
                    segment["component_count"]
                ),
                "top3": prediction["top3"],
            })

        sequence = "".join(item["digit"] for item in digits)

        uncertain_positions = [
            item["position"]
            for item in digits
            if item["is_uncertain"]
        ]

        warnings = [
            (
                "Grouped geometry-based segmentation is experimental. "
                "Touching or overlapping digits may be merged."
            ),
            (
                "Accuracy on independently collected natural handwriting "
                "has not yet been established."
            ),
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
            "digit_count": len(digits),
            "digits": digits,
            "warnings": warnings,
        }
