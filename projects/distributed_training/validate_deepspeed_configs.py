"""Validate JSON syntax and the global-batch relationship without DeepSpeed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def validate(path: Path, world_size: int) -> None:
    config = json.loads(path.read_text(encoding="utf-8"))
    micro = int(config["train_micro_batch_size_per_gpu"])
    accumulation = int(config["gradient_accumulation_steps"])
    declared = int(config["train_batch_size"])
    expected = micro * accumulation * world_size
    stage = int(config.get("zero_optimization", {}).get("stage", 0))
    if declared != expected:
        raise ValueError(
            f"{path.name}: train_batch_size={declared}, expected={expected} "
            f"for micro={micro}, accumulation={accumulation}, world_size={world_size}"
        )
    print(
        f"PASS {path.name}: ZeRO-{stage} micro={micro} accumulation={accumulation} "
        f"world_size={world_size} global_batch={declared}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("configs", nargs="*", type=Path)
    args = parser.parse_args()
    if args.world_size < 1:
        parser.error("--world-size must be >= 1")
    paths = args.configs or sorted(CONFIG_DIR.glob("*.json"))
    for path in paths:
        validate(path, args.world_size)


if __name__ == "__main__":
    main()

