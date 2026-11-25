import pytest

from src.v5_sequence_decoder import decode_v5


def make_hypothesis(
    hypothesis_id,
    digits,
    probabilities=None,
    strategy="original",
    geometry_score=0.0,
):
    probabilities = probabilities or [0.9] * len(digits)

    return {
        "hypothesis_id": hypothesis_id,
        "strategy": strategy,
        "geometry_score": geometry_score,
        "part_count": len(digits),
        "part_predictions": [
            {
                "digit": digit,
                "confidence": probability,
                "top5": [
                    {"digit": digit, "probability": probability},
                    {"digit": (digit + 1) % 10, "probability": 0.05},
                ],
            }
            for digit, probability in zip(digits, probabilities)
        ],
    }


def make_component(component_id, x, hypotheses):
    return {
        "component_id": component_id,
        "original_bbox": (x, 0, 20, 30),
        "image_bbox": (x, 0, 20, 30),
        "hypotheses": hypotheses,
    }


def test_decodes_sequence_from_multiple_components():
    components = [
        make_component("c1", 0, [make_hypothesis("h1", [1])]),
        make_component("c2", 30, [make_hypothesis("h2", [2])]),
    ]

    result = decode_v5(components)

    assert result["status"] == "sequence_hypotheses_generated"
    assert result["candidates"][0]["sequence"] == "12"
    assert result["candidates"][0]["digit_count"] == 2


def test_components_are_ordered_left_to_right():
    components = [
        make_component("right", 30, [make_hypothesis("h2", [2])]),
        make_component("left", 0, [make_hypothesis("h1", [1])]),
    ]

    result = decode_v5(components)

    assert result["candidates"][0]["sequence"] == "12"


def test_preserves_alternative_segmentation_hypotheses():
    components = [
        make_component(
            "c1",
            0,
            [
                make_hypothesis("original", [8], [0.8]),
                make_hypothesis(
                    "split",
                    [3, 1],
                    [0.9, 0.9],
                    strategy="vertical_split",
                    geometry_score=0.4,
                ),
            ],
        )
    ]

    result = decode_v5(components)
    sequences = {item["sequence"] for item in result["candidates"]}

    assert "8" in sequences
    assert "31" in sequences


def test_uses_top5_alternatives():
    components = [
        make_component(
            "c1",
            0,
            [make_hypothesis("h1", [9], [0.6])],
        )
    ]

    result = decode_v5(components)
    sequences = {item["sequence"] for item in result["candidates"]}

    assert "9" in sequences
    assert "0" in sequences


def test_never_automatically_accepts_candidate():
    components = [
        make_component("c1", 0, [make_hypothesis("h1", [4])])
    ]

    result = decode_v5(components)

    assert result["sequence"] is None
    assert result["selection_performed"] is False
    assert result["automatic_acceptance"] is False


def test_empty_components_return_no_components():
    result = decode_v5([])

    assert result["status"] == "no_components"
    assert result["sequence"] is None
    assert result["candidates"] == []


def test_rejects_invalid_top5_digit():
    components = [
        make_component(
            "c1",
            0,
            [make_hypothesis("h1", [4])],
        )
    ]
    components[0]["hypotheses"][0]["part_predictions"][0]["top5"][0][
        "digit"
    ] = 12

    with pytest.raises(ValueError, match="between 0 and 9"):
        decode_v5(components)


def test_rejects_invalid_probability():
    components = [
        make_component(
            "c1",
            0,
            [make_hypothesis("h1", [4])],
        )
    ]
    components[0]["hypotheses"][0]["part_predictions"][0]["top5"][0][
        "probability"
    ] = 1.5

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        decode_v5(components)


def test_rejects_invalid_beam_width():
    with pytest.raises(ValueError, match="beam_width"):
        decode_v5([], beam_width=0)


def test_reports_overlapping_component_order_ambiguity():
    components = [
        make_component("c1", 0, [make_hypothesis("h1", [1])]),
        make_component("c2", 10, [make_hypothesis("h2", [2])]),
    ]

    result = decode_v5(components)

    assert result["ordering_ambiguity"] is True
