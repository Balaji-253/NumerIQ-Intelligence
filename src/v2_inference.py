from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.model import DigitCNN
from src.preprocess_v2 import direct_normalize
from src.vision import enhance_image


DEFAULT_V2_CHECKPOINT = (
    "artifacts/experiments/cnn_robustness_v2/"
    "training/best_model.pt"
)


class V2Predictor:
    def __init__(
        self,
        checkpoint_path: str = DEFAULT_V2_CHECKPOINT,
        confidence_threshold: float = 0.70,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_path = Path(checkpoint_path)
        self.confidence_threshold = confidence_threshold

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing V2 checkpoint: {self.checkpoint_path}"
            )

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
            )

        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid V2 checkpoint format.")

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "V2 checkpoint has no model_state_dict."
            )

        self.model = DigitCNN().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.model_metadata = {
            "model_name": checkpoint.get(
                "model_name", "DigitCNN-Robustness-V2"
            ),
            "model_version": "cnn-robustness-v2",
            "checkpoint": str(self.checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "validation_accuracy": checkpoint.get(
                "validation_accuracy",
                checkpoint.get("accuracy"),
            ),
            "device": str(self.device),
            "preprocessing": "direct_mass_centered_v2",
        }

    def _format_prediction(
        self,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> dict:
        confidence = float(values[0].item())

        return {
            "digit": int(indices[0].item()),
            "confidence": confidence,
            "is_uncertain": (
                confidence < self.confidence_threshold
            ),
            "top3": [
                {
                    "digit": int(index.item()),
                    "probability": float(value.item()),
                }
                for value, index in zip(values[:3], indices[:3])
            ],
            "top5": [
                {
                    "digit": int(index.item()),
                    "probability": float(value.item()),
                }
                for value, index in zip(values, indices)
            ],
            "model": self.model_metadata["model_name"],
            "inference_device": str(self.device),
        }

    @torch.inference_mode()
    def predict_crops(
        self,
        crops: list[np.ndarray],
    ) -> list[dict]:
        if not crops:
            return []

        if len(crops) > 128:
            raise ValueError("Maximum batch size is 128.")

        start_time = time.perf_counter()

        tensors = [
            direct_normalize(crop, center_mass=True)
            for crop in crops
        ]

        batch = torch.cat(tensors, dim=0).to(self.device)

        logits = self.model(batch)
        probabilities = torch.softmax(logits, dim=1)

        values, indices = torch.topk(
            probabilities,
            k=5,
            dim=1,
        )

        results = [
            self._format_prediction(row_values, row_indices)
            for row_values, row_indices in zip(values, indices)
        ]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        for result in results:
            result["batch_latency_ms"] = elapsed_ms
            result["batch_size"] = len(crops)

        return results

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image.")

        start_time = time.perf_counter()

        processed = enhance_image(image)
        crop = processed["binary"]

        tensor = direct_normalize(
            crop,
            center_mass=True,
        ).to(self.device)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)

        values, indices = torch.topk(
            probabilities[0],
            k=5,
        )

        result = self._format_prediction(values, indices)
        result["latency_ms"] = (
            time.perf_counter() - start_time
        ) * 1000

        return result
