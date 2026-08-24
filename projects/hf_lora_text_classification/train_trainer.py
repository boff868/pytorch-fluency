"""Repeat the same task with Hugging Face Trainer."""

from __future__ import annotations

from transformers import Trainer, TrainingArguments

from config import parse_config
from data_pipeline import build_tokenized_datasets, build_tokenizer
from modeling import build_trainable_model, count_parameters
from utils import choose_device, save_config, seed_everything


def compute_metrics(evaluation_prediction) -> dict[str, float]:
    predictions, labels = evaluation_prediction
    predicted_labels = predictions.argmax(axis=-1)
    return {"accuracy": float((predicted_labels == labels).mean())}


def main() -> None:
    config = parse_config()
    if config.resume_model is not None:
        raise ValueError(
            "Trainer checkpoints and custom adapter resumes are different; "
            "omit --resume-model for this learning script"
        )
    save_config(config)
    seed_everything(config.seed)
    requested_device = choose_device(config.device)
    print(f"requested_device={requested_device}; Trainer performs final placement")
    tokenizer = build_tokenizer(config)
    datasets = build_tokenized_datasets(config, tokenizer)
    model = build_trainable_model(config)
    total, trainable = count_parameters(model)
    print(
        f"total_parameters={total:,} trainable_parameters={trainable:,} "
        f"ratio={100 * trainable / total:.4f}%"
    )

    training_arguments = TrainingArguments(
        output_dir=str(config.output_dir / "trainer_checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=config.seed,
        report_to=["wandb"] if config.wandb_project else [],
        run_name=config.output_dir.name if config.wandb_project else None,
        use_cpu=config.device == "cpu",
        deepspeed=(
            str(config.deepspeed_config) if config.deepspeed_config else None
        ),
    )
    if config.wandb_project:
        import os

        os.environ["WANDB_PROJECT"] = config.wandb_project
        os.environ["WANDB_MODE"] = config.wandb_mode

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(config.output_dir / "best_model"))
    tokenizer.save_pretrained(config.output_dir / "best_model")
    print(trainer.evaluate())


if __name__ == "__main__":
    main()
