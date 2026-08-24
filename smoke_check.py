"""Run all non-training labs plus a short end-to-end training smoke test."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(relative_path: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / relative_path), *args]
    print("RUN", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pytorch_fluency_smoke_") as temp:
        temp_root = Path(temp)
        run("labs/01_tensor_autograd.py")
        run("labs/02_build_network.py")
        run("labs/04_debug_lab.py")
        run("labs/05_cnn_shapes.py")
        run(
            "labs/03_train_classifier.py",
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--checkpoint",
            str(temp_root / "spiral.pt"),
        )
        cnn_output = temp_root / "cnn"
        run(
            "projects/engineered_cnn/train.py",
            "--device",
            "cpu",
            "--epochs",
            "4",
            "--lr",
            "0.003",
            "--train-samples",
            "300",
            "--val-samples",
            "90",
            "--output-dir",
            str(cnn_output),
        )
        run(
            "projects/engineered_cnn/predict.py",
            "--checkpoint",
            str(cnn_output / "best.pt"),
            "--device",
            "cpu",
            "--samples",
            "6",
        )
        gpt_output = temp_root / "mini_gpt"
        run(
            "projects/mini_gpt/train.py",
            "--device",
            "cpu",
            "--max-steps",
            "10",
            "--eval-interval",
            "5",
            "--eval-batches",
            "1",
            "--batch-size",
            "8",
            "--block-size",
            "32",
            "--embed-dim",
            "32",
            "--num-heads",
            "4",
            "--num-layers",
            "1",
            "--output-dir",
            str(gpt_output),
        )
        run(
            "projects/mini_gpt/generate.py",
            "--checkpoint",
            str(gpt_output / "best.pt"),
            "--device",
            "cpu",
            "--prompt",
            "模型",
            "--max-new-tokens",
            "8",
        )
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
