import numpy as np
import pytest

from src.segmentation_v3 import (
    extract_components_v3,
    find_vertical_split_candidates,
    create_split_hypotheses,
    segment_v3,
)


def test_empty_image_returns_no_components():
    image = np.zeros((100, 200), dtype=np.uint8)

    result = segment_v3(image)

    assert result["component_count"] == 0
    assert result["candidate_split_count"] == 0


def test_separated_components_are_detected():
    image = np.zeros((80, 160), dtype=np.uint8)
    image[15:65, 15:35] = 255
    image[15:65, 90:110] = 255

    components = extract_components_v3(image)

    assert len(components) == 2


def test_single_component_is_preserved():
    image = np.zeros((80, 80), dtype=np.uint8)
    image[10:70, 20:55] = 255

    components = extract_components_v3(image)

    assert len(components) == 1
    assert components[0]["image"].size > 0


def test_narrow_component_has_no_split_candidates():
    image = np.zeros((50, 20), dtype=np.uint8)
    image[5:45, 5:15] = 255

    candidates = find_vertical_split_candidates(image)

    assert candidates == []


def test_two_block_shapes_produce_possible_valley():
    image = np.zeros((60, 70), dtype=np.uint8)
    image[5:55, 5:25] = 255
    image[5:55, 45:65] = 255

    candidates = find_vertical_split_candidates(image)

    assert any(item["x"] in range(25, 46) for item in candidates)


def test_split_hypotheses_preserve_both_sides():
    component = {
        "image": np.pad(
            np.ones((40, 20), dtype=np.uint8) * 255,
            ((0, 0), (0, 20)),
        )
    }

    hypotheses = create_split_hypotheses(component)

    for hypothesis in hypotheses:
        assert np.any(hypothesis["left"])
        assert np.any(hypothesis["right"])


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):
        extract_components_v3(np.zeros((10, 10, 3), dtype=np.uint8))
