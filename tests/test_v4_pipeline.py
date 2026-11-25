import numpy as np
from PIL import Image

from src.v4_pipeline import V4Pipeline


class FakePredictor:
    model_metadata = {"model_name": "FakeCNN-V4-Test"}

    def predict_crops(self, crops):
        return [
            {
                "digit": 7,
                "confidence": 0.91,
                "is_uncertain": False,
                "top3": [{"digit": 7, "probability": 0.91}],
                "top5": [],
            }
            for _ in crops
        ]


def make_image():
    image = np.full((120, 240), 255, dtype=np.uint8)

    # Two separated, simple dark foreground shapes.
    image[30:85, 35:65] = 0
    image[30:85, 100:130] = 0

    return Image.fromarray(image, mode="L")


def test_pipeline_returns_expected_structure():
    pipeline = V4Pipeline(predictor=FakePredictor())

    result = pipeline.analyze(make_image())

    assert result["engine"] == "numeriq-v4"
    assert result["pipeline_version"] == "4.0.0-experimental"
    assert result["recognition_status"] == "provisional_only"
    assert result["sequence"] is None
    assert result["digit_count"] is None


def test_pipeline_preserves_uncertainty_policy():
    pipeline = V4Pipeline(predictor=FakePredictor())

    result = pipeline.analyze(make_image())

    assert result["selection"]["performed"] is False
    assert result["selection"]["automatic_acceptance"] is False
    assert result["digit_count_is_reliable"] is False


def test_pipeline_detects_empty_image():
    pipeline = V4Pipeline(predictor=FakePredictor())

    image = Image.fromarray(
        np.full((100, 100), 255, dtype=np.uint8)
    )

    result = pipeline.analyze(image)

    assert result["status"] == "no_foreground"
    assert result["segmentation"]["component_count"] == 0
    assert result["scored_components"] == []


def test_pipeline_keeps_preprocessing_metadata():
    pipeline = V4Pipeline(predictor=FakePredictor())

    result = pipeline.analyze(make_image())

    assert result["preprocessing"]["version"] == "v3"
    assert "foreground_ratio" in result["preprocessing"]
    assert "orientation" in result["preprocessing"]


def test_pipeline_keeps_component_hypotheses():
    pipeline = V4Pipeline(predictor=FakePredictor())

    result = pipeline.analyze(make_image())

    assert len(result["component_hypotheses"]) >= 1
    assert len(result["scored_components"]) >= 1

    for component in result["scored_components"]:
        assert component["selected_hypothesis_id"] is None
        assert component["automatic_acceptance"] is False


def test_pipeline_includes_limitations():
    pipeline = V4Pipeline(predictor=FakePredictor())

    result = pipeline.analyze(make_image())

    assert any(
        "Connected components" in warning
        for warning in result["warnings"]
    )
    assert any(
        "Vertical split" in warning
        for warning in result["warnings"]
    )


def test_pipeline_does_not_modify_input_image():
    image = make_image()
    original = np.asarray(image).copy()

    pipeline = V4Pipeline(predictor=FakePredictor())
    pipeline.analyze(image)

    assert np.array_equal(np.asarray(image), original)



def test_pipeline_preserves_global_coordinates_in_both_outputs():
    pipeline = V4Pipeline(predictor=FakePredictor())
    result = pipeline.analyze(make_image())

    generated = result["component_hypotheses"]
    scored = result["scored_components"]

    assert len(generated) == len(scored)
    assert len(generated) == 2

    for generated_component, scored_component in zip(generated, scored):
        bbox = generated_component["image_bbox"]

        assert bbox is not None
        assert len(bbox) == 4
        assert bbox[2] > 0
        assert bbox[3] > 0
        assert scored_component["image_bbox"] == bbox
