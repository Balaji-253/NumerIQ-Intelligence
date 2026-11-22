from __future__ import annotations

import time
from pathlib import Path

import torch

from PIL import Image

from src.model import DigitCNN
from src.preprocess import image_to_tensor


class Predictor:
    def __init__(
        self,
        checkpoint_path: str = "artifacts/mnist_cnn.pt",
        confidence_threshold: float = 0.70,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.checkpoint_path = Path(checkpoint_path)
        self.confidence_threshold = confidence_threshold

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing model checkpoint: {self.checkpoint_path}"
            )

        self.model = DigitCNN().to(self.device)

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

        if "model_state_dict" not in checkpoint:
            raise ValueError("Invalid checkpoint format.")

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.model_metadata = {
            "model_name": checkpoint.get("model_name", "DigitCNN"),
            "epoch": checkpoint.get("epoch"),
            "validation_accuracy": checkpoint.get(
                "validation_accuracy",
                checkpoint.get("accuracy"),
            ),
            "device": str(self.device),
        }

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image.")

        tensor = image_to_tensor(image)

        return tensor.to(self.device)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        start_time = time.perf_counter()

        tensor = self._prepare_image(image)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        top_values, top_indices = torch.topk(
            probabilities,
            k=5,
        )

        confidence = float(top_values[0].item())

        result = {
            "digit": int(top_indices[0].item()),
            "confidence": confidence,
            "is_uncertain": confidence < self.confidence_threshold,
            "top3": [
                {
                    "digit": int(index.item()),
                    "probability": float(value.item()),
                }
                for value, index in zip(
                    top_values[:3],
                    top_indices[:3],
                )
            ],
            "top5": [
                {
                    "digit": int(index.item()),
                    "probability": float(value.item()),
                }
                for value, index in zip(
                    top_values,
                    top_indices,
                )
            ],
            "model": self.model_metadata["model_name"],
            "inference_device": str(self.device),
            "latency_ms": (
                time.perf_counter() - start_time
            ) * 1000,
        }

        return result

    @torch.inference_mode()
    def predict_batch(self, images: list[Image.Image]) -> list[dict]:
        if not images:
            return []

        if len(images) > 128:
            raise ValueError("Maximum batch size is 128.")

        start_time = time.perf_counter()

        tensors = [
            self._prepare_image(image)
            for image in images
        ]

        batch = torch.cat(tensors, dim=0)

        logits = self.model(batch)
        probabilities = torch.softmax(logits, dim=1)

        values, indices = torch.topk(
            probabilities,
            k=5,
            dim=1,
        )

        results = []

        for row_values, row_indices in zip(values, indices):
            confidence = float(row_values[0].item())

            results.append({
                "digit": int(row_indices[0].item()),
                "confidence": confidence,
                "is_uncertain": (
                    confidence < self.confidence_threshold
                ),
                "top3": [
                    {
                        "digit": int(index.item()),
                        "probability": float(value.item()),
                    }
                    for value, index in zip(
                        row_values[:3],
                        row_indices[:3],
                    )
                ],
                "top5": [
                    {
                        "digit": int(index.item()),
                        "probability": float(value.item()),
                    }
                    for value, index in zip(
                        row_values,
                        row_indices,
                    )
                ],
                "model": self.model_metadata["model_name"],
                "inference_device": str(self.device),
            })

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        for result in results:
            result["batch_latency_ms"] = elapsed_ms
            result["batch_size"] = len(images)

        return results
