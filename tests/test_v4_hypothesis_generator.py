import numpy as np
import pytest

from src.v4_hypothesis_generator import (
    HypothesisConfig,
    generate_component_hypotheses,
    generate_image_hypotheses,
    hypotheses_to_dict,
)


def create_connected_component():
    image = np.zeros((40, 50), dtype=np.uint8)
    image[5:35, 2:20] = 255
    image[5:35, 30:48] = 255
    image[19:22, 20:30] = 255
    return image


def test_original_component_is_preserved():
    result = generate_component_hypotheses(
        create_connected_component(), 1
    )

    assert result.hypotheses
    assert result.hypotheses[0].is_original
    assert result.hypotheses[0].strategy == "original"
    assert result.hypotheses[0].cut_positions == ()


def test_hypotheses_are_never_automatically_accepted():
    result = generate_component_hypotheses(
        create_connected_component(), 1
    )

    assert all(
        not hypothesis.automatically_accepted
        for hypothesis in result.hypotheses
    )


def test_single_cut_hypotheses_are_generated():
    result = generate_component_hypotheses(
        create_connected_component(), 1
    )

    single_cuts = [
        item for item in result.hypotheses
        if item.strategy == "single_cut"
    ]

    assert len(single_cuts) >= 1
    assert all(len(item.parts) == 2 for item in single_cuts)


def test_two_cut_hypotheses_can_be_generated():
    result = generate_component_hypotheses(
        create_connected_component(),
        1,
        config=HypothesisConfig(max_cuts_per_component=2),
    )

    multiple_cuts = [
        item for item in result.hypotheses
        if item.strategy == "multiple_cut"
    ]

    assert all(len(item.parts) == 3 for item in multiple_cuts)


def test_original_only_mode():
    result = generate_component_hypotheses(
        create_connected_component(),
        1,
        config=HypothesisConfig(max_cuts_per_component=0),
    )

    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].is_original


def test_empty_component_returns_no_hypotheses():
    result = generate_component_hypotheses(
        np.zeros((20, 20), dtype=np.uint8), 1
    )

    assert result.hypotheses == []
    assert result.original_bbox == (0, 0, 0, 0)


def test_invalid_component_id_is_rejected():
    with pytest.raises(ValueError):
        generate_component_hypotheses(
            create_connected_component(), 0
        )


def test_image_hypotheses_preserve_component_traceability():
    image = np.zeros((64, 128), dtype=np.uint8)
    image[15:45, 10:25] = 255
    image[15:45, 45:60] = 255

    results = generate_image_hypotheses(image)

    assert len(results) == 2
    assert all(item.component_id > 0 for item in results)


def test_metadata_serialization_excludes_image_arrays():
    result = generate_component_hypotheses(
        create_connected_component(), 1
    )

    serialized = hypotheses_to_dict(result)

    assert serialized["component_id"] == 1
    assert serialized["hypotheses"]
    assert "foreground" not in str(serialized)


def test_hypothesis_limit_is_respected():
    result = generate_component_hypotheses(
        create_connected_component(),
        1,
        config=HypothesisConfig(max_hypotheses_per_component=2),
    )

    assert len(result.hypotheses) <= 2



def test_image_hypotheses_preserve_global_component_coordinates():
    image = np.zeros((64, 128), dtype=np.uint8)
    image[15:45, 10:25] = 255
    image[15:45, 45:60] = 255

    results = generate_image_hypotheses(image)

    assert len(results) == 2
    assert results[0].image_bbox == (10, 15, 15, 30)
    assert results[1].image_bbox == (45, 15, 15, 30)
