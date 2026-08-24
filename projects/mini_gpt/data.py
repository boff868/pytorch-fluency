"""Character tokenizer and next-token datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class CharTokenizer:
    def __init__(self, text: str | None = None, *, vocab: list[str] | None = None) -> None:
        if (text is None) == (vocab is None):
            raise ValueError("provide exactly one of text or vocab")
        self.vocab = sorted(set(text)) if text is not None else list(vocab or [])
        self.char_to_id = {character: index for index, character in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> torch.Tensor:
        unknown = sorted(set(text) - set(self.char_to_id))
        if unknown:
            raise ValueError(f"prompt contains characters outside vocabulary: {unknown}")
        return torch.tensor([self.char_to_id[character] for character in text], dtype=torch.long)

    def decode(self, token_ids: torch.Tensor | list[int]) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return "".join(self.vocab[index] for index in token_ids)


class NextTokenDataset(Dataset):
    """Return x=tokens[i:i+T] and y=tokens[i+1:i+T+1]."""

    def __init__(self, tokens: torch.Tensor, block_size: int) -> None:
        if tokens.ndim != 1 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a 1D torch.long tensor")
        if len(tokens) <= block_size:
            raise ValueError("token split must be longer than block_size")
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[index : index + self.block_size]
        y = self.tokens[index + 1 : index + self.block_size + 1]
        return x, y


def build_loaders(
    corpus_path: Path,
    block_size: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, CharTokenizer, torch.Generator]:
    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    tokens = tokenizer.encode(text)
    split_index = int(0.9 * len(tokens))
    train_tokens = tokens[:split_index]
    val_tokens = tokens[split_index:]
    train_dataset = NextTokenDataset(train_tokens, block_size)
    val_dataset = NextTokenDataset(val_tokens, block_size)
    train_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader, tokenizer, train_generator

