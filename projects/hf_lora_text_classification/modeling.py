"""Build a full-finetune or LoRA sequence classifier."""

from __future__ import annotations

from pathlib import Path

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from config import TrainConfig


def build_base_model(config: TrainConfig) -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )


def build_trainable_model(config: TrainConfig) -> PreTrainedModel:
    if config.resume_model is not None:
        if config.use_lora:
            base_model = build_base_model(config)
            return PeftModel.from_pretrained(
                base_model,
                config.resume_model,
                is_trainable=True,
            )
        return AutoModelForSequenceClassification.from_pretrained(
            config.resume_model,
        )

    model = build_base_model(config)
    if not config.use_lora:
        return model
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
    )
    return get_peft_model(model, lora_config)


def count_parameters(model: PreTrainedModel) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def load_for_inference(
    checkpoint: Path,
    base_model_id: str,
    revision: str,
) -> PreTrainedModel:
    if (checkpoint / "adapter_config.json").exists():
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_id,
            revision=revision,
            num_labels=2,
        )
        return PeftModel.from_pretrained(base_model, checkpoint)
    return AutoModelForSequenceClassification.from_pretrained(checkpoint)

