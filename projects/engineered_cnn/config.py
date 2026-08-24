"""Command-line configuration for the engineered CNN project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    device: str
    num_workers: int
    train_samples: int
    val_samples: int
    image_size: int
    hidden_channels: int
    num_classes: int
    output_dir: Path
    resume: Path | None
    deterministic: bool
    run_overfit_check: bool
    wandb_project: str | None
    wandb_mode: str


def parse_config() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the modular synthetic CNN")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=900)
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "default",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--skip-overfit-check", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.image_size < 8:
        parser.error("--image-size must be >= 8")
    if args.num_classes != 3:
        parser.error("the bundled dataset currently has exactly 3 classes")

    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        image_size=args.image_size,
        hidden_channels=args.hidden_channels,
        num_classes=args.num_classes,
        output_dir=args.output_dir.resolve(),
        resume=args.resume.resolve() if args.resume else None,
        deterministic=not args.non_deterministic,
        run_overfit_check=not args.skip_overfit_check,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
    )
