import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.preprocess_v3 import (
    preprocess_v3,
    to_grayscale,
    validate_image,
)


def create_handwriting(background, foreground):
    image = Image.new("RGB", (200, 100), background)
    draw = ImageDraw.Draw(image)
    draw.line((20, 20, 170, 75), fill=foreground, width=7)
    return image


def test_white_background_black_ink():
    result = preprocess_v3(create_handwriting("white", "black"))

    assert result["background"] == "light"
    assert result["binary"].shape == (100, 200)
    assert result["foreground_ratio"] > 0
    assert result["preprocessing_version"] == "v3"


def test_black_background_white_ink():
    result = preprocess_v3(create_handwriting("black", "white"))

    assert result["background"] == "dark"
    assert result["binary"].shape == (100, 200)
    assert result["foreground_ratio"] > 0


def test_blank_image_returns_empty_foreground():
    result = preprocess_v3(Image.new("RGB", (100, 100), "white"))

    assert result["foreground_ratio"] == 0
    assert np.count_nonzero(result["binary"]) == 0
    assert result["orientation"]["available"] is False


def test_orientation_diagnostics_are_available_for_stroke():
    result = preprocess_v3(create_handwriting("white", "black"))

    assert result["orientation"]["available"] is True
    assert isinstance(
        result["orientation"]["dominant_axis_angle_degrees"],
        float,
    )


def test_grayscale_output_is_two_dimensional():
    grayscale = to_grayscale(create_handwriting("white", "black"))

    assert grayscale.ndim == 2
    assert grayscale.dtype == np.uint8


def test_rejects_too_small_image():
    with pytest.raises(ValueError):
        validate_image(Image.new("RGB", (4, 4), "white"))


def test_rejects_non_pil_input():
    with pytest.raises(TypeError):
        preprocess_v3(np.zeros((100, 100), dtype=np.uint8))
