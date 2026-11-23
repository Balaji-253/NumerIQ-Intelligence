from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from src.preprocess_v3 import preprocess_v3
from src.segmentation_v3 import segment_v3
from src.v2_inference import V2Predictor


class V3SequenceRecognizer:
    """
    Experimental V3 sequence recognizer.

    Uses V3 preprocessing and segmentation hypotheses with the
    existing V2 CNN classifier.

    Split hypotheses are evaluated but never automatically accepted.
    """

    def __init__(self, predictor: V2Predictor):
        self.predictor = predictor

    @staticmethod
    def _empty_result(status: str, warning: str) -> dict[str, Any]:
        return {
            "status": status,
            "sequence": "",
            "digit_count": 0,
            "digits": [],
            "digit_count_is_reliable": status == "no_digits_detected",
            "segmentation_version": "v3",
            "warnings": [warning],
        }

    def recognize(self, image: Image.Image) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image.")

        processed = preprocess_v3(image)
        binary = processed["binary"]

        segmentation = segment_v3(binary)
        components = segmentation["components"]

        if not components:
            return self._empty_result(
                "no_digits_detected",
                "No foreground components were detected.",
            )

        crops: list[np.ndarray] = []
        crop_map: list[tuple[int, str, int | None]] = []

        for component_index, component in enumerate(components):
            crops.append(component["image"])
            crop_map.append((component_index, "original", None))

            for split_index, hypothesis in enumerate(
                component["split_candidates"]
            ):
                crops.append(hypothesis["left"])
                crop_map.append((component_index, "left", split_index))

                crops.append(hypothesis["right"])
                crop_map.append((component_index, "right", split_index))

        predictions = self.predictor.predict_crops(crops)

        if len(predictions) != len(crops):
            return self._empty_result(
                "recognition_failed",
                "The classifier returned an unexpected number of predictions.",
            )

        original_predictions: dict[int, dict] = {}
        split_predictions: dict[int, list[dict]] = {
            index: [] for index in range(len(components))
        }

        for mapping, prediction in zip(crop_map, predictions):
            component_index, crop_type, split_index = mapping

            if crop_type == "original":
                original_predictions[component_index] = prediction
            else:
                split_predictions[component_index].append({
                    "split_index": split_index,
                    "crop_type": crop_type,
                    "digit": str(prediction["digit"]),
                    "confidence": float(prediction["confidence"]),
                    "is_uncertain": bool(prediction["is_uncertain"]),
                    "top3": prediction["top3"],
                })

        digits = []
        uncertain_positions = []
        ambiguous_components = []

        # Experimental screening threshold; validate on real handwriting.
        MIN_MERGED_COMPONENT_ASPECT_RATIO = 0.80

        for position, component in enumerate(components):
            prediction = original_predictions[position]

            x, y, width, height = component["bbox"]
            has_split_candidates = bool(component["split_candidates"])
            aspect_ratio = float(component["aspect_ratio"])

            possible_multi_digit_component = (
                has_split_candidates
                and aspect_ratio >= MIN_MERGED_COMPONENT_ASPECT_RATIO
            )

            uncertain = (
                bool(prediction["is_uncertain"])
                or possible_multi_digit_component
            )

            if uncertain:
                uncertain_positions.append(position)

            if possible_multi_digit_component:
                ambiguous_components.append(position)

            candidate_splits = []

            for hypothesis_index, hypothesis in enumerate(
                component["split_candidates"]
            ):
                candidate_splits.append({
                    "split_x": int(hypothesis["split_x"]),
                    "left_bbox": [
                        x,
                        y,
                        int(hypothesis["split_x"]),
                        height,
                    ],
                    "right_bbox": [
                        x + int(hypothesis["split_x"]),
                        y,
                        width - int(hypothesis["split_x"]),
                        height,
                    ],
                    "valley_depth": float(hypothesis["valley_depth"]),
                    "valley_pixels": int(hypothesis["valley_pixels"]),
                    "predictions": [
                        item
                        for item in split_predictions[position]
                        if item["split_index"] == hypothesis_index
                    ],
                    "accepted": False,
                })

            digits.append({
                "position": position,
                "digit": str(prediction["digit"]),
                "confidence": float(prediction["confidence"]),
                "is_uncertain": uncertain,
                "bbox": [int(value) for value in component["bbox"]],
                "area": int(component["area"]),
                "aspect_ratio": aspect_ratio,
                "top3": prediction["top3"],
                "possible_multi_digit_component": possible_multi_digit_component,
                "candidate_splits": candidate_splits,
            })

        provisional_sequence = "".join(item["digit"] for item in digits)
        has_unresolved_components = bool(ambiguous_components)
        has_uncertainty = bool(uncertain_positions)

        # Never present an uncertain recognition as a definitive sequence.
        sequence = (
            ""
            if has_uncertainty
            else provisional_sequence
        )

        warnings = [
            (
                "V3 segmentation is experimental. Connected components "
                "do not necessarily correspond to individual digits."
            ),
            (
                "Candidate splits are diagnostic hypotheses and have "
                "not been verified as actual digit boundaries."
            ),
            (
                "CNN softmax confidence is not a calibrated probability "
                "of recognition correctness."
            ),
        ]

        if ambiguous_components:
            warnings.append(
                "Potentially merged components at positions: "
                + ", ".join(map(str, ambiguous_components))
                + ". The provisional sequence uses unsplit components and must not be treated as a verified result."
            )

        if uncertain_positions:
            warnings.append(
                "Uncertain predictions at positions: "
                + ", ".join(map(str, uncertain_positions))
            )

        return {
            "status": (
                "completed_with_uncertainty"
                if has_uncertainty or has_unresolved_components
                else "completed"
            ),
            "sequence": sequence,
            "provisional_sequence": provisional_sequence,
            "digit_count": len(digits),
            # Component count is not verified handwritten digit count.
            "digit_count_is_reliable": False,
            "digits": digits,
            "component_count": len(components),
            "candidate_split_count": segmentation[
                "candidate_split_count"
            ],
            "preprocessing": {
                "version": processed["preprocessing_version"],
                "background": processed["background"],
                "foreground_ratio": float(processed["foreground_ratio"]),
                "orientation": processed["orientation"],
            },
            "model": self.predictor.model_metadata,
            "warnings": warnings,
        }
