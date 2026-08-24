"""Check research-toolchain source and local-only executable pieces.

This does not download Hugging Face models or install optional dependencies.
Use --two-process to also launch the CPU/Gloo DDP teaching run.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HF_ROOT = ROOT / "projects" / "hf_lora_text_classification"
DDP_ROOT = ROOT / "projects" / "distributed_training"


def check_python_sources() -> None:
    files = sorted(HF_ROOT.glob("*.py")) + sorted(DDP_ROOT.glob("*.py"))
    for path in files:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
    print(f"PASS syntax: {len(files)} research-toolchain Python files")


def check_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"text", "label"}:
        raise ValueError(f"invalid CSV schema: {path}")
    labels = {int(row["label"]) for row in rows}
    if labels != {0, 1}:
        raise ValueError(f"expected labels 0/1 in {path}, got {labels}")
    print(f"PASS data: {path.name} rows={len(rows)} labels={sorted(labels)}")


def check_deepspeed_configs(world_size: int = 2) -> None:
    for path in sorted((DDP_ROOT / "configs").glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        micro = int(config["train_micro_batch_size_per_gpu"])
        accumulation = int(config["gradient_accumulation_steps"])
        declared = int(config["train_batch_size"])
        expected = micro * accumulation * world_size
        if declared != expected:
            raise ValueError(f"{path.name}: {declared=} != {expected=}")
        stage = config["zero_optimization"]["stage"]
        print(f"PASS config: {path.name} ZeRO-{stage} global_batch={declared}")


def available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_ddp(two_process: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="ddp_toolchain_check_") as temp:
        checkpoint = Path(temp) / "model.pt"
        script = str(DDP_ROOT / "ddp_train.py")
        common = [script, "--epochs", "1", "--samples", "128", "--output", str(checkpoint)]
        if two_process:
            port = str(available_port())
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=2",
                "--master_addr=127.0.0.1",
                f"--master_port={port}",
                *common,
            ]
        else:
            command = [sys.executable, *common]
        subprocess.run(command, check=True)
        if not checkpoint.exists():
            raise RuntimeError("DDP check did not create rank-zero checkpoint")
        mode = "two-process" if two_process else "single-process"
        print(f"PASS executable: DDP {mode}")


def show_optional_dependencies() -> None:
    names = ("transformers", "datasets", "peft", "accelerate", "wandb", "deepspeed")
    for name in names:
        status = "installed" if find_spec(name) else "not installed (optional)"
        print(f"DEPENDENCY {name:12} {status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-process", action="store_true")
    args = parser.parse_args()
    check_python_sources()
    check_csv(HF_ROOT / "data" / "train.csv")
    check_csv(HF_ROOT / "data" / "validation.csv")
    check_deepspeed_configs(world_size=2)
    run_ddp(args.two_process)
    show_optional_dependencies()
    print("ALL RESEARCH TOOLCHAIN LOCAL CHECKS PASSED")


if __name__ == "__main__":
    main()

