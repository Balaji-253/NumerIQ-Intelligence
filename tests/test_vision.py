import numpy as np
from PIL import Image, ImageDraw

from src.vision import enhance_image, segment_digits


def create_test_image(background="black"):
    image = Image.new(
        "RGB",
        (300, 120),
        background,
    )

    draw = ImageDraw.Draw(image)

    foreground = (
        "white" if background == "black" else "black"
    )

    # Three separated digit-like foreground components.
    draw.rectangle((35, 25, 65, 95), fill=foreground)
    draw.rectangle((125, 25, 155, 95), fill=foreground)
    draw.rectangle((215, 25, 245, 95), fill=foreground)

    return image


def test_image_enhancement():
    image = create_test_image()

    result = enhance_image(image)

    assert result["grayscale"].shape == (120, 300)
    assert result["enhanced"].shape == (120, 300)
    assert result["binary"].shape == (120, 300)

    assert set(np.unique(result["binary"])).issubset({0, 255})


def test_segment_separated_components():
    image = create_test_image()

    segments = segment_digits(image)

    assert len(segments) == 3

    x_positions = [
        segment["bbox"][0]
        for segment in segments
    ]

    assert x_positions == sorted(x_positions)

    for segment in segments:
        assert segment["image"].size > 0
        assert segment["area"] > 0


def test_white_background_segmentation():
    image = create_test_image(background="white")

    segments = segment_digits(image)

    assert len(segments) == 3


def test_empty_image_returns_no_components():
    image = Image.new("RGB", (200, 100), "black")

    segments = segment_digits(image)

    assert segments == []
