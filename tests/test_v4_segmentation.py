import numpy as np
import pytest

from src.v4_segmentation import (
    V4SegmentationConfig,
    analyze_binary_image,
    estimate_orientation,
    find_split_candidates,
)


def test_blank_image_has_no_foreground():
    image = np.zeros((64, 128), dtype=np.uint8)

    result = analyze_binary_image(image)

    assert result.status == "no_foreground"
    assert result.component_count == 0
    assert result.components == []


def test_single_component_is_extracted():
    image = np.zeros((64, 128), dtype=np.uint8)
    image[15:45, 20:40] = 255

    result = analyze_binary_image(image)

    assert result.status == "analysis_completed"
    assert result.component_count == 1
    assert result.components[0].area == 600
    assert result.components[0].bbox == (20, 15, 20, 30)


def test_separated_components_are_extracted():
    image = np.zeros((64, 128), dtype=np.uint8)
    image[15:45, 10:25] = 255
    image[15:45, 45:60] = 255

    result = analyze_binary_image(image)

    assert result.component_count == 2
    assert result.components[0].bbox[0] < result.components[1].bbox[0]


def test_touching_regions_remain_one_original_component():
    image = np.zeros((64, 128), dtype=np.uint8)
    image[15:45, 10:30] = 255
    image[15:45, 30:50] = 255

    result = analyze_binary_image(image)

    assert result.component_count == 1
    assert result.components[0].bbox == (10, 15, 40, 30)


def test_vertical_valley_produces_split_hypothesis():
    component = np.zeros((40, 50), dtype=np.uint8)
    component[5:35, 2:20] = 255
    component[5:35, 30:48] = 255
    component[19:22, 20:30] = 255

    candidates = find_split_candidates(component)

    assert len(candidates) >= 1
    assert all(0 < item.x < component.shape[1] for item in candidates)
    assert all(item.valley_depth >= 0.25 for item in candidates)
    assert all(item.left_mass > 0 for item in candidates)
    assert all(item.right_mass > 0 for item in candidates)


def test_diagonal_orientation_is_diagnostic():
    image = np.zeros((100, 100), dtype=np.uint8)

    for x in range(15, 75):
        y = x
        image[y - 2:y + 3, x - 2:x + 3] = 255

    angle = estimate_orientation(image)

    assert angle is not None
    assert 25 <= angle <= 65


def test_small_component_is_marked_as_noise():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[20, 20] = 255

    result = analyze_binary_image(image)

    assert result.component_count == 1
    assert result.noise_component_count == 1
    assert result.components[0].is_noise is True


def test_invalid_image_dimensions_are_rejected():
    with pytest.raises(ValueError):
        analyze_binary_image(np.zeros((10, 10, 3), dtype=np.uint8))


def test_non_numpy_input_is_rejected():
    with pytest.raises(TypeError):
        analyze_binary_image([[0, 255], [255, 0]])


def test_custom_configuration_is_supported():
    image = np.zeros((40, 40), dtype=np.uint8)
    image[5:35, 5:35] = 255

    config = V4SegmentationConfig(min_component_area=5)
    result = analyze_binary_image(image, config)

    assert result.component_count == 1
    assert result.noise_component_count == 0
