"""Fine-tune a pretrained classifier with an explicit PyTorch loop."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from transformers import PreTrainedModel

from config import parse_config
from data_pipeline import (
    assert_batch_contract,
    build_loaders,
    build_tokenized_datasets,
    build_tokenizer,
)
from modeling import build_trainable_model, count_parameters
from utils import (
    ExperimentTracker,
    choose_device,
    create_logger,
    save_config,
    seed_everything,
)


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def train_one_epoch(
    model: PreTrainedModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    accumulation_steps: int,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        batch = move_batch(batch, device)
        outputs = model(**batch)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite or missing loss: {loss}")
        (loss / accumulation_steps).backward()
        should_update = (
            (batch_index + 1) % accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        if should_update:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = batch["labels"].shape[0]
        loss_sum += loss.detach().item() * batch_size
        correct += (
            outputs.logits.detach().argmax(dim=1) == batch["labels"]
        ).sum().item()
        sample_count += batch_size
    return {"loss": loss_sum / sample_count, "accuracy": correct / sample_count}


@torch.inference_mode()
def evaluate(
    model: PreTrainedModel,
    loader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(**batch)
        batch_size = batch["labels"].shape[0]
        loss_sum += outputs.loss.item() * batch_size
        correct += (outputs.logits.argmax(dim=1) == batch["labels"]).sum().item()
        sample_count += batch_size
    return {"loss": loss_sum / sample_count, "accuracy": correct / sample_count}


def save_model_state(
    directory: Path,
    model: PreTrainedModel,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "optimizer_state": optimizer.state_dict(),
        },
        directory / "training_state.pt",
    )


def append_metrics(path: Path, row: dict[str, float | int]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    config = parse_config()
    payload = save_config(config)
    logger = create_logger(config.output_dir)
    tracker = ExperimentTracker(config, payload)
    seed_everything(config.seed)
    device = choose_device(config.device)

    tokenizer = build_tokenizer(config)
    datasets = build_tokenized_datasets(config, tokenizer)
    train_loader, validation_loader = build_loaders(config, tokenizer, datasets)
    first_batch = next(iter(train_loader))
    assert_batch_contract(first_batch)

    model = build_trainable_model(config).to(device)
    total_parameters, trainable_parameters = count_parameters(model)
    trainable_ratio = trainable_parameters / total_parameters
    logger.info(
        "device=%s total_parameters=%d trainable_parameters=%d ratio=%.4f%%",
        device,
        total_parameters,
        trainable_parameters,
        trainable_ratio * 100,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    start_epoch = 1
    best_val_loss = float("inf")
    if config.resume_model is not None:
        state_path = config.resume_model / "training_state.pt"
        state = torch.load(state_path, map_location=device)
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        best_val_loss = float(state["best_val_loss"])
        logger.info("resumed at epoch=%d from %s", start_epoch, config.resume_model)

    try:
        for epoch in range(start_epoch, config.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                config.grad_clip,
                config.gradient_accumulation_steps,
            )
            validation_metrics = evaluate(model, validation_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": validation_metrics["loss"],
                "val_accuracy": validation_metrics["accuracy"],
                "trainable_ratio": trainable_ratio,
            }
            logger.info(
                "epoch=%d train_loss=%.4f train_acc=%.2f%% val_loss=%.4f val_acc=%.2f%%",
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"] * 100,
                validation_metrics["loss"],
                validation_metrics["accuracy"] * 100,
            )
            append_metrics(config.output_dir / "metrics.csv", row)
            tracker.log(row, step=epoch)
            improved = validation_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = validation_metrics["loss"]
            save_model_state(
                config.output_dir / "last_model",
                model,
                tokenizer,
                optimizer,
                epoch,
                best_val_loss,
            )
            if improved:
                save_model_state(
                    config.output_dir / "best_model",
                    model,
                    tokenizer,
                    optimizer,
                    epoch,
                    best_val_loss,
                )
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
