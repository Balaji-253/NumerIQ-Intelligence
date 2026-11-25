from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from src.preprocess_v3 import preprocess_v3
from src.v2_inference import V2Predictor
from src.v4_segmentation import (
    DEFAULT_CONFIG,
    V4SegmentationConfig,
    analyze_binary_image,
    analysis_to_dict,
)
from src.v4_hypothesis_generator import (
    HypothesisConfig,
    generate_image_hypotheses,
    hypotheses_to_dict,
)
from src.v4_candidate_scorer import score_component_hypotheses


class CropPredictor(Protocol):
    def predict_crops(self, crops: list) -> list[dict]:
        ...


class V4Pipeline:
    """
    Integrated experimental V4 handwriting analysis pipeline.

    This pipeline produces component-level hypotheses and CNN scores.
    It does not claim a definitive digit count or sequence.
    """

    def __init__(
        self,
        predictor: CropPredictor | None = None,
        checkpoint_path: str = (
            "artifacts/experiments/cnn_robustness_v2/"
            "training/best_model.pt"
        ),
        hypothesis_config: HypothesisConfig = HypothesisConfig(),
        segmentation_config: V4SegmentationConfig = DEFAULT_CONFIG,
    ) -> None:
        self.predictor = predictor or V2Predictor(
            checkpoint_path=str(Path(checkpoint_path))
        )
        self.hypothesis_config = hypothesis_config
        self.segmentation_config = segmentation_config

    def analyze(self, image: Image.Image) -> dict[str, Any]:
        """
        Preprocess and analyze one handwritten image.

        Original image data is not modified.
        All segmentation alternatives are retained.
        """

        # Stage 1: preprocessing.
        preprocessing = preprocess_v3(image)
        binary = preprocessing["binary"]

        # Stage 2: connected-component analysis.
        segmentation = analyze_binary_image(
            binary,
            config=self.segmentation_config,
        )

        # Stage 3: generate alternative interpretations.
        component_sets = generate_image_hypotheses(
            binary,
            config=self.hypothesis_config,
            segmentation_config=self.segmentation_config,
        )

        # Stage 4: score all candidate crops with the existing CNN.
        scored_components = score_component_hypotheses(
            component_sets,
            self.predictor,
        )

        warnings = [
            (
                "Connected components are not guaranteed to correspond "
                "to individual handwritten digits."
            ),
            (
                "Vertical split candidates are hypotheses, not verified "
                "digit boundaries."
            ),
            (
                "CNN scores are ranking heuristics and are not calibrated "
                "probabilities of correctness."
            ),
            (
                "The current pipeline does not reliably resolve arbitrary "
                "overlapping, diagonal or slanted digit boundaries."
            ),
        ]

        if preprocessing["foreground_ratio"] > 0.65:
            warnings.append(
                "High foreground ratio: image polarity or thresholding "
                "may require manual review."
            )

        if preprocessing["foreground_ratio"] < 0.001:
            warnings.append(
                "Very little foreground detected; handwriting may have "
                "been missed during preprocessing."
            )

        return {
            "engine": "numeriq-v4",
            "pipeline_version": "4.0.0-experimental",
            "status": (
                "no_foreground"
                if segmentation.status == "no_foreground"
                else "analysis_completed"
            ),
            "recognition_status": "provisional_only",
            "sequence": None,
            "digit_count": None,
            "digit_count_is_reliable": False,
            "preprocessing": {
                "version": preprocessing["preprocessing_version"],
                "background": preprocessing["background"],
                "border_median": preprocessing["border_median"],
                "foreground_ratio": preprocessing["foreground_ratio"],
                "orientation": preprocessing["orientation"],
            },
            "segmentation": analysis_to_dict(segmentation),
            "component_hypotheses": [
                hypotheses_to_dict(item)
                for item in component_sets
            ],
            "scored_components": scored_components,
            "selection": {
                "performed": False,
                "automatic_acceptance": False,
                "reason": (
                    "No validated sequence-level decoder or "
                    "segmentation acceptance policy is available."
                ),
            },
            "warnings": warnings,
            "model": getattr(
                self.predictor,
                "model_metadata",
                {"model_name": "custom predictor"},
            ),
        }
