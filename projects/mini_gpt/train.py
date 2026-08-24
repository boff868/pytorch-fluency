"""Train and evaluate a character-level MiniGPT."""

from __future__ import annotations

import json
from dataclasses import asdict

import torch
from torch import nn

from config import parse_config
from data import build_loaders
from model import GPTConfig, MiniGPT, count_trainable_parameters
from utils import (
    append_metrics,
    choose_device,
    create_logger,
    restore_training,
    save_checkpoint,
    seed_everything,
)


@torch.inference_mode()
def evaluate_loss(
    model: MiniGPT,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    loss_sum = 0.0
    batch_count = 0
    for tokens, targets in loader:
        tokens = tokens.to(device)
        targets = targets.to(device)
        _, loss = model(tokens, targets)
        assert loss is not None
        loss_sum += loss.item()
        batch_count += 1
        if batch_count >= max_batches:
            break
    if batch_count == 0:
        raise RuntimeError("evaluation loader produced no batches")
    return loss_sum / batch_count


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    config = parse_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    for key, value in payload.items():
        if hasattr(value, "as_posix"):
            payload[key] = str(value)
    (config.output_dir / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger = create_logger(config.output_dir)
    seed_everything(config.seed)
    device = choose_device(config.device)

    train_loader, val_loader, tokenizer, train_generator = build_loaders(
        config.corpus,
        config.block_size,
        config.batch_size,
        config.seed,
    )
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model = MiniGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    start_step = 0
    best_val_loss = float("inf")
    if config.resume is not None:
        start_step, best_val_loss = restore_training(
            config.resume,
            model=model,
            optimizer=optimizer,
            device=device,
            train_generator=train_generator,
        )
        logger.info("resumed from %s at completed_step=%d", config.resume, start_step)

    tokens, targets = next(iter(train_loader))
    assert tokens.shape == targets.shape
    assert tokens.ndim == 2 and tokens.shape[1] == config.block_size
    with torch.inference_mode():
        dummy_logits, _ = model(tokens[:2].to(device))
    assert dummy_logits.shape == (
        2,
        config.block_size,
        tokenizer.vocab_size,
    )

    logger.info(
        "PyTorch=%s device=%s vocab=%d parameters=%d train_windows=%d val_windows=%d",
        torch.__version__,
        device,
        tokenizer.vocab_size,
        count_trainable_parameters(model),
        len(train_loader.dataset),
        len(val_loader.dataset),
    )
    train_iterator = iter(train_loader)
    model.train()

    for step_index in range(start_step, config.max_steps):
        (tokens, targets), train_iterator = next_batch(train_iterator, train_loader)
        tokens = tokens.to(device)
        targets = targets.to(device)
        _, loss = model(tokens, targets)
        assert loss is not None and torch.isfinite(loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        completed_step = step_index + 1

        should_evaluate = (
            completed_step == 1
            or completed_step % config.eval_interval == 0
            or completed_step == config.max_steps
        )
        if should_evaluate:
            train_loss = evaluate_loss(model, train_loader, device, config.eval_batches)
            val_loss = evaluate_loss(model, val_loader, device, config.eval_batches)
            logger.info(
                "step=%05d batch_loss=%.4f train_loss=%.4f val_loss=%.4f grad_norm=%.3f",
                completed_step,
                loss.item(),
                train_loss,
                val_loss,
                float(grad_norm),
            )
            append_metrics(
                config.output_dir / "metrics.csv",
                {
                    "step": completed_step,
                    "batch_loss": loss.item(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "grad_norm": float(grad_norm),
                },
            )
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
            save_checkpoint(
                config.output_dir / "last.pt",
                step=completed_step,
                model=model,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                model_config=model_config.to_dict(),
                vocab=tokenizer.vocab,
                train_generator=train_generator,
            )
            if improved:
                save_checkpoint(
                    config.output_dir / "best.pt",
                    step=completed_step,
                    model=model,
                    optimizer=optimizer,
                    best_val_loss=best_val_loss,
                    model_config=model_config.to_dict(),
                    vocab=tokenizer.vocab,
                    train_generator=train_generator,
                )
            model.train()

    if start_step >= config.max_steps:
        logger.info("checkpoint already reached requested max_steps=%d", config.max_steps)
    logger.info("finished; best_val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()

