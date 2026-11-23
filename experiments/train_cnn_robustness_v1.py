from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model import DigitCNN


DATA_DIR = Path(
    "artifacts/experiments/cnn_robustness_v1/crops"
)

OUTPUT_DIR = Path(
    "artifacts/experiments/cnn_robustness_v1/training"
)

CHECKPOINT = OUTPUT_DIR / "best_model.pt"
LAST_CHECKPOINT = OUTPUT_DIR / "last_checkpoint.pt"

MEAN = 0.1307
STD = 0.3081


class DiskBackedDigitDataset(Dataset):
    """Read accepted image records directly from NumPy memmaps."""

    def __init__(self, partition: str):
        metadata_path = DATA_DIR / "metadata.json"

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        self.count = int(
            metadata[partition]["accepted_crops"]
        )

        self.images = np.load(
            DATA_DIR / f"{partition}_images.npy",
            mmap_mode="r",
        )

        self.labels = np.load(
            DATA_DIR / f"{partition}_labels.npy",
            mmap_mode="r",
        )

        if self.images.shape[0] < self.count:
            raise RuntimeError("Image array is smaller than accepted count.")

        if self.labels.shape[0] < self.count:
            raise RuntimeError("Label array is smaller than accepted count.")

        if self.images.shape[1:] != (28, 28):
            raise RuntimeError("Unexpected image dimensions.")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        image = torch.from_numpy(
            self.images[index].copy()
        ).unsqueeze(0).float()

        image.div_(255.0)
        image.sub_(MEAN).div_(STD)

        label = int(self.labels[index])

        return image, label


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    class_correct = torch.zeros(10, dtype=torch.long)
    class_total = torch.zeros(10, dtype=torch.long)

    confusion = torch.zeros((10, 10), dtype=torch.long)

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            correct += (predictions == labels).sum().item()
            total += batch_size

            for label, prediction in zip(labels, predictions):
                label_index = int(label.item())
                prediction_index = int(prediction.item())

                class_total[label_index] += 1
                class_correct[label_index] += int(
                    label_index == prediction_index
                )

                confusion[label_index, prediction_index] += 1

    per_class_accuracy = {}

    for digit in range(10):
        denominator = int(class_total[digit])

        per_class_accuracy[str(digit)] = (
            int(class_correct[digit]) / denominator
            if denominator else None
        )

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
    }


def save_checkpoint(path, checkpoint):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(path.name + ".tmp")

    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.set_num_threads(args.threads)

    device = torch.device("cpu")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NUMERIQ CNN ROBUSTNESS V1 TRAINING")
    print("=" * 70)
    print("Device:", device)
    print("Architecture: DigitCNN")
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("CPU threads:", args.threads)
    print("Learning rate:", args.learning_rate)
    print("Training data: disk-backed inference-matched crops")
    print("Original checkpoint: protected")

    train_dataset = DiskBackedDigitDataset("train")
    validation_dataset = DiskBackedDigitDataset("validation")

    print("\nTraining records:", len(train_dataset))
    print("Validation records:", len(validation_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = DigitCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_accuracy = -1.0
    best_epoch = 0
    history = []

    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        epoch_started = time.perf_counter()

        for batch_index, (images, labels) in enumerate(
            train_loader,
            start=1,
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total += batch_size

            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Batch {batch_index}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}",
                    flush=True,
                )

        train_loss = running_loss / total
        train_accuracy = correct / total

        validation = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        scheduler.step(validation["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.perf_counter() - epoch_started

        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "learning_rate": current_lr,
            "epoch_seconds": epoch_time,
        }

        history.append(result)

        print("\n" + "=" * 65)
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"Training loss: {train_loss:.5f}")
        print(f"Training accuracy: {train_accuracy:.2%}")
        print(f"Validation loss: {validation['loss']:.5f}")
        print(f"Validation accuracy: {validation['accuracy']:.2%}")
        print(f"Learning rate: {current_lr:.7f}")
        print(f"Epoch time: {epoch_time:.1f} seconds")
        print("=" * 65, flush=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "model_name": "DigitCNN",
            "experiment": "cnn_robustness_v1",
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "validation_accuracy": validation["accuracy"],
            "validation_loss": validation["loss"],
            "per_class_accuracy": validation["per_class_accuracy"],
            "confusion_matrix": validation["confusion_matrix"],
            "seed": args.seed,
            "normalization": {
                "mean": MEAN,
                "std": STD,
            },
            "training_records": len(train_dataset),
            "validation_records": len(validation_dataset),
        }

        save_checkpoint(LAST_CHECKPOINT, checkpoint)

        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch

            save_checkpoint(CHECKPOINT, checkpoint)

            print(
                f"Best checkpoint saved: {CHECKPOINT}",
                flush=True,
            )

        history_path = OUTPUT_DIR / "training_history.json"

        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

    elapsed = time.perf_counter() - started

    summary = {
        "experiment": "NumerIQ CNN Robustness V1",
        "epochs_completed": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "training_records": len(train_dataset),
        "validation_records": len(validation_dataset),
        "batch_size": args.batch_size,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "best_checkpoint": str(CHECKPOINT),
        "original_checkpoint_modified": False,
        "official_test_used": False,
        "history": history,
    }

    summary_path = OUTPUT_DIR / "training_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 70)
    print("CNN ROBUSTNESS V1 TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Best epoch: {best_epoch}")
    print(f"Total time: {elapsed / 60:.2f} minutes")
    print(f"Best checkpoint: {CHECKPOINT}")
    print(f"Summary: {summary_path}")
    print("Official MNIST test set was not used.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)

    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("Epochs and batch size must be positive.")

    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Invalid optimizer parameters.")

    if args.threads <= 0:
        parser.error("Threads must be positive.")

    return args


if __name__ == "__main__":
    main(parse_args())
