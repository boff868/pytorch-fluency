"""Configuration for MiniGPT training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainConfig:
    corpus: Path
    output_dir: Path
    resume: Path | None
    device: str
    seed: int
    max_steps: int
    eval_interval: int
    eval_batches: int
    batch_size: int
    block_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    grad_clip: float


def parse_config() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a character-level MiniGPT")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "corpus.txt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "default")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps must be >= 1")
    if args.eval_interval < 1:
        parser.error("--eval-interval must be >= 1")
    if args.embed_dim % args.num_heads != 0:
        parser.error("--embed-dim must be divisible by --num-heads")
    if args.block_size < 2:
        parser.error("--block-size must be >= 2")

    return TrainConfig(
        corpus=args.corpus.resolve(),
        output_dir=args.output_dir.resolve(),
        resume=args.resume.resolve() if args.resume else None,
        device=args.device,
        seed=args.seed,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )

