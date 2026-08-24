"""Reload a full model or LoRA adapter and classify text."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modeling import load_for_inference
from utils import choose_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model-id", default="distilbert-base-uncased")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--text", action="append")
    args = parser.parse_args()
    texts = args.text or [
        "the tutorial is clear and useful",
        "the experiment is confusing and unreliable",
    ]

    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = load_for_inference(
        args.checkpoint,
        args.base_model_id,
        args.model_revision,
    ).to(device)
    model.eval()
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    with torch.inference_mode():
        probabilities = model(**batch).logits.softmax(dim=1).cpu()

    labels = ("negative", "positive")
    for text, distribution in zip(texts, probabilities):
        predicted = int(distribution.argmax().item())
        print(
            f"label={labels[predicted]:8} confidence={distribution[predicted]:.1%} "
            f"text={text}"
        )


if __name__ == "__main__":
    main()

