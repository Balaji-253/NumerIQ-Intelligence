from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from src.v4_hypothesis_generator import ComponentHypothesisSet


class CropPredictor(Protocol):
    def predict_crops(
        self,
        crops: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        ...


def score_component_hypotheses(
    component_sets: list[ComponentHypothesisSet],
    predictor: CropPredictor,
    batch_size: int = 128,
) -> list[dict[str, Any]]:
    """
    Score every V4 segmentation hypothesis using the existing CNN.

    This function ranks hypotheses heuristically. It does not select,
    accept, or verify a segmentation.

    CNN confidence is not a calibrated probability of correctness.
    Geometry scores are retained separately and are not combined with
    CNN scores into an invented probability.
    """
    if not 1 <= batch_size <= 128:
        raise ValueError("batch_size must be between 1 and 128.")

    # Each entry identifies the exact hypothesis part being classified.
    crop_references: list[tuple[int, int, int]] = []
    crops: list[np.ndarray] = []

    for component_index, component_set in enumerate(component_sets):
        for hypothesis_index, hypothesis in enumerate(
            component_set.hypotheses
        ):
            for part_index, part in enumerate(hypothesis.parts):
                crop = part.foreground

                if not isinstance(crop, np.ndarray):
                    raise TypeError("Hypothesis crop must be a NumPy array.")

                if crop.ndim != 2 or crop.size == 0:
                    raise ValueError("Hypothesis crop must be a nonempty 2D array.")

                if not np.any(crop > 0):
                    raise ValueError("Hypothesis crop contains no foreground.")

                crops.append(crop.astype(np.uint8, copy=False))
                crop_references.append(
                    (component_index, hypothesis_index, part_index)
                )

    # Run the existing V2 model in bounded batches.
    predictions_by_reference: dict[tuple[int, int, int], dict] = {}

    for start in range(0, len(crops), batch_size):
        batch_crops = crops[start:start + batch_size]
        batch_predictions = predictor.predict_crops(batch_crops)

        if len(batch_predictions) != len(batch_crops):
            raise RuntimeError(
                "Predictor returned a different number of predictions "
                "than input crops."
            )

        for offset, prediction in enumerate(batch_predictions):
            if not isinstance(prediction, dict):
                raise TypeError("Each prediction must be a dictionary.")

            try:
                digit = int(prediction["digit"])
                confidence = float(prediction["confidence"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Prediction must contain a valid digit and confidence."
                ) from exc

            if not 0 <= digit <= 9:
                raise ValueError("Predicted digit must be between 0 and 9.")

            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("Prediction confidence must be in [0, 1].")

            reference = crop_references[start + offset]
            predictions_by_reference[reference] = prediction

    # Reconstruct every hypothesis without discarding alternatives.
    results: list[dict[str, Any]] = []

    for component_index, component_set in enumerate(component_sets):
        scored_hypotheses = []

        for hypothesis_index, hypothesis in enumerate(
            component_set.hypotheses
        ):
            part_predictions = []
            log_scores = []

            for part_index, part in enumerate(hypothesis.parts):
                prediction = predictions_by_reference[
                    (component_index, hypothesis_index, part_index)
                ]

                confidence = float(prediction["confidence"])
                log_scores.append(math.log(max(confidence, 1e-12)))

                part_predictions.append(
                    {
                        "part_index": part.part_index,
                        "bbox": part.bbox,
                        "area": part.area,
                        "digit": int(prediction["digit"]),
                        "confidence": confidence,
                        "is_uncertain": bool(
                            prediction.get("is_uncertain", False)
                        ),
                        "top3": prediction.get("top3", []),
                        "top5": prediction.get("top5", []),
                    }
                )

            if log_scores:
                mean_log_score = sum(log_scores) / len(log_scores)
                geometric_mean = math.exp(mean_log_score)
            else:
                mean_log_score = None
                geometric_mean = None

            provisional_sequence = "".join(
                str(item["digit"]) for item in part_predictions
            )

            scored_hypotheses.append(
                {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "component_id": hypothesis.component_id,
                    "strategy": hypothesis.strategy,
                    "cut_positions": list(hypothesis.cut_positions),
                    "is_original": hypothesis.is_original,
                    "automatically_accepted": False,
                    "geometry_score": float(hypothesis.geometry_score),
                    "part_count": len(hypothesis.parts),
                    "provisional_sequence": provisional_sequence,
                    "part_predictions": part_predictions,
                    "mean_log_top1_score": mean_log_score,
                    "geometric_mean_top1_score": geometric_mean,
                    "ranking_basis": (
                        "mean_log_top1_score"
                        if mean_log_score is not None
                        else "not_scored"
                    ),
                    "rank": None,
                }
            )

        # Rank only as a diagnostic aid. This is not a decision.
        rankable = [
            item for item in scored_hypotheses
            if item["mean_log_top1_score"] is not None
        ]
        rankable.sort(
            key=lambda item: item["mean_log_top1_score"],
            reverse=True,
        )

        for rank, item in enumerate(rankable, start=1):
            item["rank"] = rank

        results.append(
            {
                "component_id": component_set.component_id,
                "original_bbox": component_set.original_bbox,
                "image_bbox": component_set.image_bbox,
                "hypothesis_count": len(scored_hypotheses),
                "hypotheses": scored_hypotheses,
                "selected_hypothesis_id": None,
                "selection_status": "not_selected",
                "automatic_acceptance": False,
            }
        )

    return results
