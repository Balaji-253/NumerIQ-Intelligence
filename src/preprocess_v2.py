from __future__ import annotations

import numpy as np
import torch
from PIL import Image

IMAGE_SIZE = 28
DIGIT_SIZE = 20
MEAN = 0.1307
STD = 0.3081


def direct_normalize(
    crop: np.ndarray,
    center_mass: bool = True,
) -> torch.Tensor:
    """Normalize a white-foreground binary crop using V2 benchmark logic."""

    if not isinstance(crop, np.ndarray) or crop.ndim != 2:
        raise ValueError("Expected a two-dimensional binary crop.")

    if crop.size == 0:
        raise ValueError("Empty crop received.")

    foreground = crop > 0

    if not foreground.any():
        raise ValueError("Crop contains no foreground pixels.")

    ys, xs = np.where(foreground)

    cropped = crop[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1,
    ]

    image = Image.fromarray(cropped.astype(np.uint8), mode="L")
    image.thumbnail(
        (DIGIT_SIZE, DIGIT_SIZE),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    left = (IMAGE_SIZE - image.width) // 2
    top = (IMAGE_SIZE - image.height) // 2
    canvas.paste(image, (left, top))

    array = np.asarray(canvas, dtype=np.float32).copy()

    if center_mass:
        total_mass = float(array.sum())

        if total_mass > 0:
            ys, xs = np.indices(array.shape)
            center_x = float((xs * array).sum() / total_mass)
            center_y = float((ys * array).sum() / total_mass)

            shift_x = int(round(13.5 - center_x))
            shift_y = int(round(13.5 - center_y))

            canvas = canvas.transform(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Transform.AFFINE,
                (1, 0, -shift_x, 0, 1, -shift_y),
                resample=Image.Resampling.BILINEAR,
                fillcolor=0,
            )

            array = np.asarray(canvas, dtype=np.float32).copy()

    array /= 255.0
    array = (array - MEAN) / STD

    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
