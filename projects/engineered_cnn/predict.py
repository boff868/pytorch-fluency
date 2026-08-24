"""Load best checkpoint and run inference on fresh synthetic samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import CLASS_NAMES, make_pattern_dataset
from model import SmallCNN
from utils import choose_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SmallCNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = make_pattern_dataset(args.samples, image_size=16, seed=999)
    images, labels = next(iter(DataLoader(dataset, batch_size=args.samples)))
    with torch.inference_mode():
        probabilities = model(images.to(device)).softmax(dim=1).cpu()
    predictions = probabilities.argmax(dim=1)

    for index, (prediction, label) in enumerate(zip(predictions, labels)):
        confidence = probabilities[index, prediction].item()
        print(
            f"sample={index:02d} true={CLASS_NAMES[label]} "
            f"pred={CLASS_NAMES[prediction]} confidence={confidence:.1%}"
        )
    accuracy = (predictions == labels).float().mean().item()
    print(f"accuracy={accuracy:.1%} checkpoint_epoch={checkpoint['epoch']}")


if __name__ == "__main__":
    main()

