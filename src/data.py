from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DEFAULT_DATA_DIR = "data"
DEFAULT_BATCH_SIZE = 128
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 0


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_loaders(
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    num_workers: int = DEFAULT_NUM_WORKERS,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    set_seed(seed)

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    transform = get_transforms()

    full_train_dataset = datasets.MNIST(
        root=str(data_path),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=str(data_path),
        train=False,
        download=True,
        transform=transform,
    )

    if len(full_train_dataset) != 60000:
        raise RuntimeError(
            f"Unexpected training dataset size: {len(full_train_dataset)}"
        )

    if len(test_dataset) != 10000:
        raise RuntimeError(
            f"Unexpected test dataset size: {len(test_dataset)}"
        )

    generator = torch.Generator().manual_seed(seed)

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [55000, 5000],
        generator=generator,
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_options,
    )

    return train_loader, validation_loader, test_loader
