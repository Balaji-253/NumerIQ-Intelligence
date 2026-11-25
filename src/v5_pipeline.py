from __future__ import annotations

from typing import Any

from PIL import Image

from src.v4_pipeline import V4Pipeline
from src.v5_sequence_decoder import decode_v5


class V5Pipeline:
    """
    Experimental V5 pipeline.

    Reuses V4 preprocessing, segmentation, hypothesis generation,
    and CNN scoring, then generates complete sequence hypotheses.
    """

    def __init__(
        self,
        v4_pipeline: V4Pipeline | None = None,
        beam_width: int = 64,
        max_results: int = 20,
    ) -> None:
        if not 1 <= beam_width <= 512:
            raise ValueError("beam_width must be between 1 and 512.")

        if not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100.")

        self.v4_pipeline = v4_pipeline or V4Pipeline()
        self.beam_width = beam_width
        self.max_results = max_results

    def analyze(self, image: Image.Image) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image.")

        # Preserve the complete V4 analysis.
        v4_result = self.v4_pipeline.analyze(image)

        scored_components = v4_result.get("scored_components", [])

        # Generate sequence alternatives without accepting one.
        decoding = decode_v5(
            scored_components=scored_components,
            beam_width=self.beam_width,
            max_results=self.max_results,
        )

        return {
            "engine": "numeriq-v5",
            "version": "5.0.0-experimental",
            "status": "analysis_completed",
            "recognition_status": "provisional_only",
            "sequence": None,
            "digit_count": None,
            "digit_count_is_reliable": False,
            "v4_analysis": v4_result,
            "sequence_decoding": decoding,
            "selection": {
                "performed": False,
                "automatic_acceptance": False,
                "reason": (
                    "V5 sequence ranking is experimental and has not "
                    "been validated as a reliable recognition decision."
                ),
            },
            "warnings": [
                "V5 reuses the existing V2 CNN; it is not a newly trained model.",
                "Sequence candidates are ranked using uncalibrated CNN outputs.",
                "Alternative segmentation hypotheses may contain incorrect cuts.",
                "The highest-ranked sequence is not guaranteed to be correct.",
                "Real-world validation is required before definitive recognition.",
            ],
            "model": v4_result.get("model"),
        }
