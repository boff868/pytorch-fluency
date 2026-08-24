"""实验 3：一个完整、无外部数据依赖的三分类训练项目。

Shape contract:
    x:      [B, 2], float32
    logits: [B, 3], float32
    y:      [B],    int64, values in [0, 2]
    loss:   scalar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def make_spiral(
    samples_per_class: int = 400,
    num_classes: int = 3,
    noise: float = 0.18,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a learnable 2D spiral classification dataset."""
    generator = torch.Generator().manual_seed(seed)
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for class_index in range(num_classes):
        radius = torch.linspace(0.05, 1.0, samples_per_class)
        angle = torch.linspace(
            class_index * 4.0,
            (class_index + 1) * 4.0,
            samples_per_class,
        )
        angle += torch.randn(samples_per_class, generator=generator) * noise
        points = torch.stack(
            (radius * torch.sin(angle), radius * torch.cos(angle)),
            dim=1,
        )
        features.append(points)
        labels.append(
            torch.full((samples_per_class,), class_index, dtype=torch.long)
        )

    x = torch.cat(features).to(torch.float32)
    y = torch.cat(labels)
    permutation = torch.randperm(len(y), generator=generator)
    return x[permutation], y[permutation]


class SpiralMLP(nn.Module):
    """Contract: [B, 2] -> [B, 3] logits."""

    def __init__(self, hidden: int = 64, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_batch_contract(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
) -> None:
    assert x.ndim == 2 and x.shape[1] == 2, x.shape
    assert y.ndim == 1 and y.shape[0] == x.shape[0], y.shape
    assert x.dtype == torch.float32, x.dtype
    assert y.dtype == torch.long, y.dtype
    assert torch.isfinite(x).all()
    assert y.min().item() >= 0
    assert y.max().item() < num_classes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        assert logits.shape == (x.shape[0], 3), logits.shape
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = y.shape[0]
        loss_sum += loss.detach().item() * batch_size
        correct += (logits.detach().argmax(dim=1) == y).sum().item()
        sample_count += batch_size

    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            batch_size = y.shape[0]
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == y).sum().item()
            sample_count += batch_size

    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
    }


def overfit_one_batch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    steps: int = 120,
) -> float:
    """Pipeline unit test: a capable model should memorize a tiny batch."""
    x, y = next(iter(loader))
    x = x[:32].to(device)
    y = y[:32].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)
    model.train()

    for _ in range(steps):
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        accuracy = (model(x).argmax(dim=1) == y).float().mean().item()
    return accuracy


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "torch_version": torch.__version__,
            "model_config": {"hidden": 64, "num_classes": 3},
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "best_spiral.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"PyTorch {torch.__version__} | device={device}")

    x, y = make_spiral(seed=args.seed)
    dataset = TensorDataset(x, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    first_x, first_y = next(iter(train_loader))
    check_batch_contract(first_x, first_y, num_classes=3)
    print(f"first batch: x={tuple(first_x.shape)}, y={tuple(first_y.shape)}")

    # Run the one-batch diagnostic on a disposable model, not the final model.
    probe_model = SpiralMLP().to(device)
    probe_accuracy = overfit_one_batch(
        probe_model,
        train_loader,
        nn.CrossEntropyLoss(),
        device,
    )
    print(f"one-batch overfit accuracy={probe_accuracy:.1%}")
    if probe_accuracy < 0.95:
        raise RuntimeError("one-batch overfit check failed")

    # Reset seeds so the diagnostic above does not affect the real run.
    torch.manual_seed(args.seed)
    model = SpiralMLP().to(device)
    dummy_logits = model(first_x[:2].to(device))
    assert dummy_logits.shape == (2, 3)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        print(
            f"epoch={epoch:02d} "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.1%} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.1%}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                args.checkpoint,
                epoch,
                model,
                optimizer,
                best_val_loss,
            )

    # Verify that a fresh model loaded from disk gives the same logits.
    model.eval()
    sample = first_x[:5].to(device)
    with torch.inference_mode():
        logits_before_load = model(sample).cpu()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    restored_model = SpiralMLP(**checkpoint["model_config"]).to(device)
    restored_model.load_state_dict(checkpoint["model_state"])
    restored_model.eval()
    with torch.inference_mode():
        logits_after_load = restored_model(sample).cpu()

    # The saved checkpoint may be from an earlier best epoch, so compare the
    # restored model with the saved state loaded into the current model as well.
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.inference_mode():
        logits_from_saved_state = model(sample).cpu()
    assert torch.allclose(logits_from_saved_state, logits_after_load, atol=1e-6)

    print(
        f"checkpoint={args.checkpoint} | best epoch={checkpoint['epoch']} | "
        f"best val loss={checkpoint['best_val_loss']:.4f}"
    )
    print(
        "last-vs-best logits identical:",
        torch.allclose(logits_before_load, logits_after_load, atol=1e-6),
        "(False is normal if the best epoch was earlier)",
    )
    print("PASS: full training/checkpoint lab")


if __name__ == "__main__":
    main()

