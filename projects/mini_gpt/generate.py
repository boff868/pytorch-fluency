"""Generate text from a trained MiniGPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import CharTokenizer
from model import GPTConfig, MiniGPT
from utils import choose_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="模型")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    tokenizer = CharTokenizer(vocab=checkpoint["vocab"])
    model = MiniGPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    prompt_tokens = tokenizer.encode(args.prompt).unsqueeze(0).to(device)
    generated = model.generate(
        prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(generated[0]))


if __name__ == "__main__":
    main()

