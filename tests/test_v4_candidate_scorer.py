import numpy as np
import pytest

from src.v4_candidate_scorer import score_component_hypotheses
from src.v4_hypothesis_generator import (
    ComponentHypothesisSet,
    HypothesisPart,
    SegmentationHypothesis,
)


class FakePredictor:
    def __init__(self, predictions=None):
        self.predictions = predictions or []
        self.calls = []

    def predict_crops(self, crops):
        self.calls.append(len(crops))
        start = sum(self.calls[:-1])
        return self.predictions[start:start + len(crops)]


def make_part(index=0):
    return HypothesisPart(
        part_index=index,
        bbox=(0, 0, 5, 8),
        area=20,
        foreground=np.full((8, 5), 255, dtype=np.uint8),
    )


def make_hypothesis(hypothesis_id, confidence, original=False):
    return SegmentationHypothesis(
        hypothesis_id=hypothesis_id,
        component_id=1,
        strategy="original" if original else "single_cut",
        cut_positions=(),
        parts=[make_part()],
        geometry_score=0.0 if original else 0.5,
        is_original=original,
    )


def prediction(digit, confidence):
    return {
        "digit": digit,
        "confidence": confidence,
        "is_uncertain": confidence < 0.70,
        "top3": [{"digit": digit, "probability": confidence}],
        "top5": [],
    }


def test_preserves_original_and_alternative_hypotheses():
    hypotheses = [
        make_hypothesis("original", 0.9, original=True),
        make_hypothesis("split", 0.8),
    ]
    component = ComponentHypothesisSet(1, (0, 0, 10, 10), hypotheses)
    predictor = FakePredictor([
        prediction(8, 0.9),
        prediction(3, 0.8),
    ])

    result = score_component_hypotheses([component], predictor)[0]

    assert result["hypothesis_count"] == 2
    assert {item["hypothesis_id"] for item in result["hypotheses"]} == {
        "original", "split"
    }
    assert result["selected_hypothesis_id"] is None


def test_ranks_hypotheses_by_mean_log_score():
    hypotheses = [
        make_hypothesis("lower", 0.6),
        make_hypothesis("higher", 0.9),
    ]
    component = ComponentHypothesisSet(1, (0, 0, 10, 10), hypotheses)
    predictor = FakePredictor([
        prediction(2, 0.6),
        prediction(7, 0.9),
    ])

    result = score_component_hypotheses([component], predictor)[0]
    ranked = sorted(
        result["hypotheses"],
        key=lambda item: item["rank"],
    )

    assert ranked[0]["hypothesis_id"] == "higher"
    assert ranked[1]["hypothesis_id"] == "lower"


def test_never_automatically_accepts_hypothesis():
    component = ComponentHypothesisSet(
        1, (0, 0, 10, 10),
        [make_hypothesis("original", 0.99, original=True)],
    )
    predictor = FakePredictor([prediction(5, 0.99)])

    result = score_component_hypotheses([component], predictor)[0]

    assert result["automatic_acceptance"] is False
    assert result["selected_hypothesis_id"] is None
    assert result["hypotheses"][0]["automatically_accepted"] is False


def test_empty_component_list_returns_empty_result():
    predictor = FakePredictor()

    assert score_component_hypotheses([], predictor) == []
    assert predictor.calls == []


def test_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        score_component_hypotheses([], FakePredictor(), batch_size=129)


def test_rejects_invalid_prediction_confidence():
    component = ComponentHypothesisSet(
        1, (0, 0, 10, 10),
        [make_hypothesis("original", 0.5, original=True)],
    )
    predictor = FakePredictor([prediction(5, 1.5)])

    with pytest.raises(ValueError, match="confidence"):
        score_component_hypotheses([component], predictor)


def test_batches_more_than_128_crops():
    hypotheses = [
        SegmentationHypothesis(
            hypothesis_id=f"h{i}",
            component_id=1,
            strategy="original",
            cut_positions=(),
            parts=[make_part()],
            geometry_score=0.0,
            is_original=True,
        )
        for i in range(130)
    ]
    component = ComponentHypothesisSet(1, (0, 0, 10, 10), hypotheses)
    predictor = FakePredictor([
        prediction(i % 10, 0.8) for i in range(130)
    ])

    result = score_component_hypotheses([component], predictor)[0]

    assert predictor.calls == [128, 2]
    assert result["hypothesis_count"] == 130


def test_rejects_empty_foreground_crop():
    empty_part = HypothesisPart(
        part_index=0,
        bbox=(0, 0, 0, 0),
        area=0,
        foreground=np.zeros((0, 0), dtype=np.uint8),
    )
    hypothesis = SegmentationHypothesis(
        hypothesis_id="empty",
        component_id=1,
        strategy="original",
        cut_positions=(),
        parts=[empty_part],
        geometry_score=0.0,
        is_original=True,
    )
    component = ComponentHypothesisSet(
        1, (0, 0, 0, 0), [hypothesis]
    )

    with pytest.raises(ValueError):
        score_component_hypotheses([component], FakePredictor())



def test_scorer_preserves_global_component_coordinates():
    component = ComponentHypothesisSet(
        1,
        (0, 0, 10, 10),
        [make_hypothesis("original", 0.9, original=True)],
        image_bbox=(25, 12, 10, 10),
    )
    predictor = FakePredictor([prediction(7, 0.9)])

    result = score_component_hypotheses([component], predictor)[0]

    assert result["image_bbox"] == (25, 12, 10, 10)


def test_scorer_supports_legacy_components_without_global_coordinates():
    component = ComponentHypothesisSet(
        1,
        (0, 0, 10, 10),
        [make_hypothesis("original", 0.9, original=True)],
    )
    predictor = FakePredictor([prediction(7, 0.9)])

    result = score_component_hypotheses([component], predictor)[0]

    assert result["image_bbox"] is None
