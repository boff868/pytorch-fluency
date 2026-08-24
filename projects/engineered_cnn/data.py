"""Synthetic image data and DataLoader construction."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = ("vertical", "horizontal", "diagonal")


def make_pattern_dataset(
    num_samples: int,
    image_size: int,
    seed: int,
    noise_std: float = 0.20,
) -> TensorDataset:
    """Create noisy vertical, horizontal, and diagonal line images."""
    generator = torch.Generator().manual_seed(seed)
    images = torch.zeros(num_samples, 1, image_size, image_size)
    labels = torch.arange(num_samples, dtype=torch.long) % len(CLASS_NAMES)
    center = image_size // 2

    for index, label_tensor in enumerate(labels):
        label = int(label_tensor.item())
        shift = int(torch.randint(-2, 3, (1,), generator=generator).item())
        image = images[index, 0]
        if label == 0:
            column = max(1, min(image_size - 2, center + shift))
            image[:, column - 1 : column + 1] = 1.0
        elif label == 1:
            row = max(1, min(image_size - 2, center + shift))
            image[row - 1 : row + 1, :] = 1.0
        else:
            diagonal = torch.arange(image_size)
            shifted = (diagonal + shift).clamp(0, image_size - 1)
            image[diagonal, shifted] = 1.0
            image[diagonal, (shifted + 1).clamp(max=image_size - 1)] = 1.0

    noise = torch.randn(images.shape, generator=generator) * noise_std
    images = (images + noise).clamp(0.0, 1.0).to(torch.float32)
    permutation = torch.randperm(num_samples, generator=generator)
    return TensorDataset(images[permutation], labels[permutation])


def build_loaders(
    train_samples: int,
    val_samples: int,
    image_size: int,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    train_dataset = make_pattern_dataset(train_samples, image_size, seed)
    val_dataset = make_pattern_dataset(val_samples, image_size, seed + 1)
    train_generator = torch.Generator().manual_seed(seed + 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, train_generator


def assert_batch_contract(
    images: torch.Tensor,
    labels: torch.Tensor,
    image_size: int,
    num_classes: int,
) -> None:
    assert images.ndim == 4
    assert images.shape[1:] == (1, image_size, image_size)
    assert labels.shape == (images.shape[0],)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.long
    assert torch.isfinite(images).all()
    assert labels.min().item() >= 0
    assert labels.max().item() < num_classes

