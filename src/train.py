from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data import get_loaders, set_seed
from src.model import DigitCNN


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def train(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("NUMERIQ INTELLIGENCE - MODEL TRAINING")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Random seed: {args.seed}")

    train_loader, val_loader, _ = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("Test samples: reserved for final evaluation only")

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

    best_val_accuracy = -1.0
    best_epoch = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

        train_loss = running_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch}/{args.epochs}"
            f"\nTrain Loss: {train_loss:.4f}"
            f" | Train Accuracy: {train_accuracy:.2%}"
            f"\nValidation Loss: {val_loss:.4f}"
            f" | Validation Accuracy: {val_accuracy:.2%}"
            f"\nLearning Rate: {current_lr:.7f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "accuracy": val_accuracy,
                "validation_accuracy": val_accuracy,
                "validation_loss": val_loss,
                "train_accuracy": train_accuracy,
                "train_loss": train_loss,
                "seed": args.seed,
                "model_name": "DigitCNN",
            }

            temporary_path = output_path.with_name(
                output_path.name + ".tmp"
            )

            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, output_path)

            print(f"Best validation checkpoint saved: {output_path}")

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Best epoch: {best_epoch}")
    print(f"Checkpoint: {output_path}")
    print("Independent test set has not been used.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the NumerIQ handwritten digit classifier."
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/mnist_cnn.pt",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero.")

    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative.")

    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")

    return args


if __name__ == "__main__":
    train(parse_args())
