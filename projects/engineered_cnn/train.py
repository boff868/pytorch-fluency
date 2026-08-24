"""Composition root for the engineered CNN training project."""

from __future__ import annotations

import torch
from torch import nn

from checkpoint import restore_training, save_checkpoint
from config import parse_config
from data import assert_batch_contract, build_loaders
from engine import evaluate, overfit_one_batch, train_one_epoch
from model import SmallCNN, count_trainable_parameters
from utils import (
    ExperimentTracker,
    append_metrics,
    choose_device,
    create_logger,
    save_config,
    seed_everything,
)


def main() -> None:
    config = parse_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, config.output_dir / "config.json")
    logger = create_logger(config.output_dir)
    tracker = ExperimentTracker(
        config.wandb_project,
        config.wandb_mode,
        config,
        config.output_dir,
    )
    seed_everything(config.seed, config.deterministic)
    device = choose_device(config.device)

    train_loader, val_loader, train_generator = build_loaders(
        train_samples=config.train_samples,
        val_samples=config.val_samples,
        image_size=config.image_size,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    first_images, first_labels = next(iter(train_loader))
    assert_batch_contract(
        first_images,
        first_labels,
        config.image_size,
        config.num_classes,
    )

    model_config = {
        "hidden_channels": config.hidden_channels,
        "num_classes": config.num_classes,
    }
    model = SmallCNN(**model_config).to(device)
    with torch.inference_mode():
        dummy_logits = model(first_images[:2].to(device))
    assert dummy_logits.shape == (2, config.num_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    start_epoch = 1
    best_val_loss = float("inf")

    if config.resume is not None:
        start_epoch, best_val_loss = restore_training(
            config.resume,
            model=model,
            optimizer=optimizer,
            device=device,
            train_generator=train_generator,
        )
        logger.info("resumed from %s at epoch %d", config.resume, start_epoch)
    elif config.run_overfit_check:
        probe = SmallCNN(**model_config).to(device)
        probe_accuracy = overfit_one_batch(probe, train_loader, device)
        logger.info("one-batch overfit accuracy=%.1f%%", probe_accuracy * 100)
        if probe_accuracy < 0.95:
            raise RuntimeError("one-batch overfit diagnostic failed")
        # Restore initial seed because the disposable diagnostic consumed RNG.
        seed_everything(config.seed, config.deterministic)
        model = SmallCNN(**model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    logger.info(
        "PyTorch=%s device=%s parameters=%d train=%d val=%d",
        torch.__version__,
        device,
        count_trainable_parameters(model),
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    try:
        for epoch in range(start_epoch, config.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val_metrics = evaluate(model, val_loader, loss_fn, device)
            logger.info(
                "epoch=%03d train_loss=%.4f train_acc=%.2f%% val_loss=%.4f val_acc=%.2f%%",
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"] * 100,
                val_metrics["loss"],
                val_metrics["accuracy"] * 100,
            )
            metric_row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            append_metrics(config.output_dir / "metrics.csv", metric_row)
            tracker.log(metric_row, step=epoch)

            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
            save_checkpoint(
                config.output_dir / "last.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                model_config=model_config,
                train_generator=train_generator,
            )
            if improved:
                save_checkpoint(
                    config.output_dir / "best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    best_val_loss=best_val_loss,
                    model_config=model_config,
                    train_generator=train_generator,
                )
    finally:
        tracker.finish()

    if start_epoch > config.epochs:
        logger.info("checkpoint already reached requested total epochs=%d", config.epochs)
    logger.info("finished; best_val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()
