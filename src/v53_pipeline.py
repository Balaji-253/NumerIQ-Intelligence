from __future__ import annotations

from typing import Any

from PIL import Image

from src.v4_pipeline import V4Pipeline
from src.v53_sequence_decoder import decode_v5


class V53Pipeline:
    """
    Experimental V5.3 pipeline.

    Reuses V4 preprocessing, segmentation, hypothesis generation,
    and CNN scoring. Uses the V5.2 decoder with segmentation-path
    diversity during final candidate selection.
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

        v4_result = self.v4_pipeline.analyze(image)

        decoding = decode_v5(
            scored_components=v4_result.get("scored_components", []),
            beam_width=self.beam_width,
            max_results=self.max_results,
        )

        return {
            "engine": "numeriq-v5.3",
            "version": "5.3.0-experimental",
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
                    "V5.3 candidate generation is experimental. "
                    "No sequence is accepted automatically."
                ),
            },
            "warnings": [
                "V5.2 reuses the existing V2 CNN; it is not a newly trained model.",
                "CNN probabilities are not calibrated correctness probabilities.",
                "Segmentation-path diversity is used during candidate selection.",
                "Alternative segmentation hypotheses may contain incorrect cuts.",
                "The highest-ranked sequence is not guaranteed to be correct.",
                "Real-world validation across independent writers is required.",
            ],
            "model": v4_result.get("model"),
        }
