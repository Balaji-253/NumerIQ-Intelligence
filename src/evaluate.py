from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data import get_loaders
from src.model import DigitCNN


def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError("Invalid checkpoint: model_state_dict missing.")

    model = DigitCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def evaluate(model, test_loader, device):
    all_predictions = []
    all_targets = []
    all_confidences = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            confidences, predictions = probabilities.max(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(labels.tolist())
            all_confidences.extend(confidences.cpu().tolist())

    return (
        np.asarray(all_targets),
        np.asarray(all_predictions),
        np.asarray(all_confidences),
    )


def save_confusion_matrix(targets, predictions, output_dir):
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(10)),
    )

    figure, axis = plt.subplots(figsize=(10, 8))

    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set(
        title="MNIST Test Confusion Matrix",
        xlabel="Predicted Digit",
        ylabel="Actual Digit",
        xticks=range(10),
        yticks=range(10),
    )

    for row in range(10):
        for column in range(10):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=8,
            )

    figure.tight_layout()
    figure.savefig(
        output_dir / "confusion_matrix.png",
        dpi=200,
    )
    plt.close(figure)

    return matrix


def save_misclassified_analysis(
    targets,
    predictions,
    confidences,
    output_dir,
):
    errors = np.where(targets != predictions)[0]

    error_records = []

    for index in errors:
        error_records.append({
            "index": int(index),
            "actual": int(targets[index]),
            "predicted": int(predictions[index]),
            "confidence": float(confidences[index]),
        })

    with open(
        output_dir / "misclassified_samples.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(error_records, file, indent=2)

    return len(errors)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("NUMERIQ INTELLIGENCE - INDEPENDENT MODEL EVALUATION")
    print("=" * 65)

    _, _, test_loader = get_loaders(
        data_dir="data",
        batch_size=128,
        seed=42,
    )

    print(f"Independent test samples: {len(test_loader.dataset)}")

    model, checkpoint = load_model(
        "artifacts/mnist_cnn.pt",
        device,
    )

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Device: {device}")

    targets, predictions, confidences = evaluate(
        model,
        test_loader,
        device,
    )

    accuracy = accuracy_score(targets, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    report = classification_report(
        targets,
        predictions,
        labels=list(range(10)),
        target_names=[str(i) for i in range(10)],
        digits=4,
        zero_division=0,
    )

    average_confidence = float(np.mean(confidences))
    error_count = int(np.sum(targets != predictions))

    metrics = {
        "evaluation_dataset": "MNIST independent test set",
        "test_samples": int(len(targets)),
        "accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "average_confidence": average_confidence,
        "error_count": error_count,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_accuracy": checkpoint.get(
            "validation_accuracy",
            checkpoint.get("accuracy"),
        ),
        "device": str(device),
    }

    matrix = save_confusion_matrix(
        targets,
        predictions,
        output_dir,
    )

    save_misclassified_analysis(
        targets,
        predictions,
        confidences,
        output_dir,
    )

    with open(
        output_dir / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    with open(
        output_dir / "evaluation_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\nCLASSIFICATION REPORT")
    print("-" * 65)
    print(report)

    print("\nOVERALL TEST METRICS")
    print("-" * 65)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nArtifacts generated:")
    print(output_dir / "classification_report.txt")
    print(output_dir / "confusion_matrix.png")
    print(output_dir / "evaluation_metrics.json")
    print(output_dir / "misclassified_samples.json")

    print("\nConfusion matrix shape:", matrix.shape)
    print("Independent evaluation completed.")


if __name__ == "__main__":
    main()
