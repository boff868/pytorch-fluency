"""List Linear module names before choosing LoRA target_modules."""

from __future__ import annotations

import argparse

from torch import nn
from transformers import AutoModelForSequenceClassification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="distilbert-base-uncased")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        revision=args.revision,
        num_labels=2,
    )
    print("Linear modules (full name -> final component):")
    final_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            final_name = name.rsplit(".", 1)[-1]
            final_names.add(final_name)
            print(f"{name:70} -> {final_name}")
    print("\nCandidate target_modules:", ",".join(sorted(final_names)))


if __name__ == "__main__":
    main()
