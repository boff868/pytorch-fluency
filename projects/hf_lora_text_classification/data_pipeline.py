"""Hugging Face Datasets and dynamic-padding DataLoaders."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase

from config import TrainConfig


def build_tokenizer(config: TrainConfig) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
    )


def build_tokenized_datasets(
    config: TrainConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetDict:
    raw = load_dataset(
        "csv",
        data_files={
            "train": str(config.train_file),
            "validation": str(config.validation_file),
        },
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_length,
        )

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized


def build_loaders(
    config: TrainConfig,
    tokenizer: PreTrainedTokenizerBase,
    datasets: DatasetDict,
) -> tuple[DataLoader, DataLoader]:
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    return train_loader, validation_loader


def assert_batch_contract(batch: dict[str, torch.Tensor]) -> None:
    required = {"input_ids", "attention_mask", "labels"}
    missing = required - set(batch)
    assert not missing, f"missing batch fields: {missing}"
    assert batch["input_ids"].ndim == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == (batch["input_ids"].shape[0],)
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["labels"].min().item() >= 0
    assert batch["labels"].max().item() < 2

