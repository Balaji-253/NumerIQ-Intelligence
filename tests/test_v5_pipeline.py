from PIL import Image
import pytest

from src.v5_pipeline import V5Pipeline


class FakeV4Pipeline:
    def analyze(self, image):
        return {
            "engine": "numeriq-v4",
            "status": "analysis_completed",
            "model": {"model_name": "test-model"},
            "scored_components": [
                {
                    "component_id": "c1",
                    "original_bbox": (0, 0, 20, 30),
                    "image_bbox": (0, 0, 20, 30),
                    "hypotheses": [
                        {
                            "hypothesis_id": "h1",
                            "strategy": "original",
                            "geometry_score": 0.0,
                            "part_count": 1,
                            "part_predictions": [
                                {
                                    "digit": 3,
                                    "confidence": 0.9,
                                    "top5": [
                                        {"digit": 3, "probability": 0.9},
                                        {"digit": 8, "probability": 0.05},
                                    ],
                                }
                            ],
                        }
                    ],
                },
                {
                    "component_id": "c2",
                    "original_bbox": (30, 0, 20, 30),
                    "image_bbox": (30, 0, 20, 30),
                    "hypotheses": [
                        {
                            "hypothesis_id": "h2",
                            "strategy": "original",
                            "geometry_score": 0.0,
                            "part_count": 1,
                            "part_predictions": [
                                {
                                    "digit": 7,
                                    "confidence": 0.8,
                                    "top5": [
                                        {"digit": 7, "probability": 0.8},
                                        {"digit": 1, "probability": 0.1},
                                    ],
                                }
                            ],
                        }
                    ],
                },
            ],
        }


def test_pipeline_generates_sequence_candidates():
    pipeline = V5Pipeline(v4_pipeline=FakeV4Pipeline())
    result = pipeline.analyze(Image.new("L", (100, 100), 255))

    assert result["engine"] == "numeriq-v5"
    assert result["recognition_status"] == "provisional_only"
    assert result["sequence"] is None
    assert result["sequence_decoding"]["candidates"][0]["sequence"] == "37"


def test_pipeline_preserves_v4_analysis():
    pipeline = V5Pipeline(v4_pipeline=FakeV4Pipeline())
    result = pipeline.analyze(Image.new("L", (100, 100), 255))

    assert result["v4_analysis"]["engine"] == "numeriq-v4"
    assert len(result["v4_analysis"]["scored_components"]) == 2


def test_pipeline_never_automatically_accepts():
    pipeline = V5Pipeline(v4_pipeline=FakeV4Pipeline())
    result = pipeline.analyze(Image.new("L", (100, 100), 255))

    assert result["selection"]["performed"] is False
    assert result["selection"]["automatic_acceptance"] is False
    assert result["sequence_decoding"]["automatic_acceptance"] is False


def test_pipeline_rejects_non_pil_input():
    pipeline = V5Pipeline(v4_pipeline=FakeV4Pipeline())

    with pytest.raises(TypeError, match="PIL Image"):
        pipeline.analyze("not an image")


def test_pipeline_rejects_invalid_beam_width():
    with pytest.raises(ValueError, match="beam_width"):
        V5Pipeline(v4_pipeline=FakeV4Pipeline(), beam_width=0)


def test_pipeline_rejects_invalid_max_results():
    with pytest.raises(ValueError, match="max_results"):
        V5Pipeline(v4_pipeline=FakeV4Pipeline(), max_results=101)
