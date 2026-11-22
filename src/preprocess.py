from __future__ import annotations

import numpy as np
import torch

from PIL import Image, ImageOps, ImageFilter


IMAGE_SIZE = 28
DIGIT_SIZE = 20
NORMALIZATION_MEAN = 0.1307
NORMALIZATION_STD = 0.3081


def _to_grayscale(image: Image.Image) -> Image.Image:
    """Convert an image safely to grayscale."""
    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    elif image.mode == "P":
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)

    return ImageOps.grayscale(image)


def _otsu_threshold(array: np.ndarray) -> int:
    """Calculate an Otsu threshold without requiring OpenCV."""
    histogram = np.bincount(
        array.ravel(),
        minlength=256,
    ).astype(np.float64)

    total = array.size
    total_sum = np.dot(np.arange(256), histogram)

    background_weight = 0.0
    background_sum = 0.0
    maximum_variance = 0.0
    threshold = 127

    for value in range(256):
        background_weight += histogram[value]

        if background_weight == 0:
            continue

        foreground_weight = total - background_weight

        if foreground_weight == 0:
            break

        background_sum += value * histogram[value]

        background_mean = background_sum / background_weight
        foreground_mean = (
            total_sum - background_sum
        ) / foreground_weight

        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )

        if variance > maximum_variance:
            maximum_variance = variance
            threshold = value

    return int(threshold)


def _detect_foreground(array: np.ndarray) -> np.ndarray:
    """
    Convert the image into a white-digit-on-black-background mask.

    Background polarity is estimated from the image borders.
    """
    height, width = array.shape

    border_width = max(1, min(height, width) // 30)

    border = np.concatenate([
        array[:border_width, :].ravel(),
        array[-border_width:, :].ravel(),
        array[:, :border_width].ravel(),
        array[:, -border_width:].ravel(),
    ])

    background_level = float(np.median(border))

    if background_level > 127:
        array = 255 - array

    threshold = _otsu_threshold(array)

    # Keep antialiased foreground while suppressing weak background noise.
    effective_threshold = max(12, int(threshold * 0.55))

    mask = array > effective_threshold

    # Remove extremely small isolated components.
    # This is a conservative pixel-level filter, not connected-component
    # segmentation.
    if mask.sum() > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        padded = np.pad(mask.astype(np.uint8), 1)

        neighbors = np.zeros_like(mask, dtype=np.uint8)

        for dy in range(3):
            for dx in range(3):
                neighbors += padded[
                    dy:dy + height,
                    dx:dx + width,
                ]

        isolated = neighbors <= 1
        mask &= ~isolated

    return np.where(mask, array, 0).astype(np.uint8)


def _crop_foreground(
    array: np.ndarray,
    padding: int = 2,
) -> np.ndarray:
    """Crop around detected foreground pixels."""
    ys, xs = np.where(array > 0)

    if len(xs) == 0:
        return array

    height, width = array.shape

    left = max(0, int(xs.min()) - padding)
    right = min(width, int(xs.max()) + padding + 1)

    top = max(0, int(ys.min()) - padding)
    bottom = min(height, int(ys.max()) + padding + 1)

    return array[top:bottom, left:right]


def _resize_preserving_aspect(
    array: np.ndarray,
) -> Image.Image:
    """Resize foreground into a 20x20 maximum region."""
    image = Image.fromarray(array, mode="L")

    image.thumbnail(
        (DIGIT_SIZE, DIGIT_SIZE),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)

    left = (IMAGE_SIZE - image.width) // 2
    top = (IMAGE_SIZE - image.height) // 2

    canvas.paste(image, (left, top))

    return canvas


def _center_of_mass(image: Image.Image) -> Image.Image:
    """Align foreground center of mass to the canvas center."""
    array = np.asarray(image, dtype=np.float32)

    total_mass = float(array.sum())

    if total_mass <= 0:
        return image

    ys, xs = np.indices(array.shape)

    center_x = float((xs * array).sum() / total_mass)
    center_y = float((ys * array).sum() / total_mass)

    target_x = (IMAGE_SIZE - 1) / 2
    target_y = (IMAGE_SIZE - 1) / 2

    shift_x = int(round(target_x - center_x))
    shift_y = int(round(target_y - center_y))

    # PIL affine transform maps output coordinates to input coordinates.
    return image.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Transform.AFFINE,
        (1, 0, -shift_x, 0, 1, -shift_y),
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Complete preprocessing pipeline.

    Output:
        28x28 grayscale image with white foreground on black background.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    if image.width <= 0 or image.height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if image.width > 4096 or image.height > 4096:
        raise ValueError("Image dimensions exceed the 4096-pixel limit.")

    image = _to_grayscale(image)

    # Mild denoising while preserving digit strokes.
    image = image.filter(ImageFilter.MedianFilter(size=3))

    array = np.asarray(image, dtype=np.uint8)

    foreground = _detect_foreground(array)

    cropped = _crop_foreground(foreground)

    normalized = _resize_preserving_aspect(cropped)

    normalized = _center_of_mass(normalized)

    return normalized


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image into a normalized MNIST-compatible tensor.

    Returns:
        Tensor with shape [1, 1, 28, 28].
    """
    processed = preprocess_image(image)

    array = np.asarray(
        processed,
        dtype=np.float32,
    ).copy() / 255.0

    tensor = torch.from_numpy(array)

    tensor = (
        tensor - NORMALIZATION_MEAN
    ) / NORMALIZATION_STD

    return tensor.unsqueeze(0).unsqueeze(0)
