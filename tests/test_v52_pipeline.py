from unittest.mock import Mock

import pytest
from PIL import Image

from src.v52_pipeline import V52Pipeline


def make_v4_result():
    return {
        "status": "analysis_completed",
        "scored_components": [],
        "model": {"name": "V2"},
    }


def test_pipeline_generates_sequence_candidates():
    v4 = Mock()
    v4.analyze.return_value = make_v4_result()

    pipeline = V52Pipeline(v4_pipeline=v4)
    result = pipeline.analyze(Image.new("L", (100, 100), 255))

    assert result["engine"] == "numeriq-v5.2"
    assert result["version"] == "5.2.0-experimental"
    assert result["status"] == "analysis_completed"
    assert result["recognition_status"] == "provisional_only"
    assert "sequence_decoding" in result
    assert result["sequence_decoding"]["engine"] == "numeriq-v5.2"


def test_pipeline_preserves_v4_analysis():
    v4 = Mock()
    v4_result = make_v4_result()
    v4.analyze.return_value = v4_result

    pipeline = V52Pipeline(v4_pipeline=v4)
    result = pipeline.analyze(Image.new("L", (100, 100), 255))

    assert result["v4_analysis"] is v4_result
    assert result["model"] == v4_result["model"]
    v4.analyze.assert_called_once()


def test_pipeline_never_automatically_accepts():
    v4 = Mock()
    v4.analyze.return_value = make_v4_result()

    pipeline = V52Pipeline(v4_pipeline=v4)
    result = pipeline.analyze(Image.new("L", (100, 100), 255))

    assert result["sequence"] is None
    assert result["digit_count"] is None
    assert result["digit_count_is_reliable"] is False
    assert result["selection"]["performed"] is False
    assert result["selection"]["automatic_acceptance"] is False
    assert result["sequence_decoding"]["automatic_acceptance"] is False


def test_pipeline_rejects_non_pil_input():
    v4 = Mock()
    pipeline = V52Pipeline(v4_pipeline=v4)

    with pytest.raises(TypeError, match="image must be a PIL Image"):
        pipeline.analyze("not an image")

    v4.analyze.assert_not_called()


def test_pipeline_rejects_invalid_beam_width():
    with pytest.raises(ValueError, match="beam_width"):
        V52Pipeline(beam_width=0)


def test_pipeline_rejects_invalid_max_results():
    with pytest.raises(ValueError, match="max_results"):
        V52Pipeline(max_results=101)


def test_pipeline_passes_decoder_limits(monkeypatch):
    import src.v52_pipeline as pipeline_module

    captured = {}

    def fake_decoder(scored_components, beam_width, max_results):
        captured["scored_components"] = scored_components
        captured["beam_width"] = beam_width
        captured["max_results"] = max_results
        return {
            "engine": "numeriq-v5.2",
            "status": "sequence_hypotheses_generated",
            "candidates": [],
            "automatic_acceptance": False,
        }

    monkeypatch.setattr(pipeline_module, "decode_v5", fake_decoder)

    v4 = Mock()
    v4.analyze.return_value = make_v4_result()

    pipeline = V52Pipeline(
        v4_pipeline=v4,
        beam_width=32,
        max_results=10,
    )
    pipeline.analyze(Image.new("L", (100, 100), 255))

    assert captured["scored_components"] == []
    assert captured["beam_width"] == 32
    assert captured["max_results"] == 10
