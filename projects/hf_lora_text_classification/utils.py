"""Local logging, reproducibility, Git metadata, and optional W&B."""

from __future__ import annotations

import json
import logging
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from config import TrainConfig


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_payload(config: TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    payload["target_modules"] = list(config.target_modules)
    payload["git_commit"] = get_git_commit()
    payload["torch_version"] = torch.__version__
    return payload


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def create_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"hf_lora.{output_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def save_config(config: TrainConfig) -> dict[str, Any]:
    payload = config_payload(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


class ExperimentTracker:
    def __init__(self, config: TrainConfig, payload: dict[str, Any]) -> None:
        self.run = None
        if config.wandb_project is None:
            return
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("install wandb or omit --wandb-project") from error
        self.run = wandb.init(
            project=config.wandb_project,
            config=payload,
            dir=str(config.output_dir),
            mode=config.wandb_mode,
        )

    def log(self, metrics: dict[str, float | int], step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

