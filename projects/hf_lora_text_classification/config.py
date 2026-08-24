"""Configuration shared by custom-loop and Trainer examples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainConfig:
    model_id: str
    model_revision: str
    train_file: Path
    validation_file: Path
    output_dir: Path
    resume_model: Path | None
    device: str
    seed: int
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    wandb_project: str | None
    wandb_mode: str
    deepspeed_config: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="distilbert-base-uncased")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--train-file", type=Path, default=PROJECT_ROOT / "data" / "train.csv")
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "validation.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "default")
    parser.add_argument("--resume-model", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_lin,v_lin")
    parser.add_argument("--wandb-project")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--deepspeed-config", type=Path)
    return parser


def parse_config() -> TrainConfig:
    parser = build_parser()
    args = parser.parse_args()
    target_modules = tuple(
        name.strip() for name in args.target_modules.split(",") if name.strip()
    )
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be >= 1")
    if not target_modules and not args.no_lora:
        parser.error("LoRA requires at least one --target-modules entry")

    return TrainConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        train_file=args.train_file.resolve(),
        validation_file=args.validation_file.resolve(),
        output_dir=args.output_dir.resolve(),
        resume_model=args.resume_model.resolve() if args.resume_model else None,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        deepspeed_config=(
            args.deepspeed_config.resolve() if args.deepspeed_config else None
        ),
    )
