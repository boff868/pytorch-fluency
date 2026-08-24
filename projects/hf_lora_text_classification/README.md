# 项目三：Hugging Face + Datasets + LoRA 文本分类

目标：用同一任务串起真实研究工具链，同时保留你已经掌握的 PyTorch 训练循环。

```text
CSV
-> datasets.load_dataset
-> tokenizer + map
-> dynamic-padding collator
-> pretrained Transformer
-> full fine-tune 或 LoRA
-> custom loop / Trainer
-> W&B + adapter/checkpoint
-> reload + predict
```

项目使用一份很小的英文正负面文本数据，便于检查流程。它不是用来追求高指标的正式 benchmark。

## 1. 独立环境

当前教程的基础环境未安装 Hugging Face 工具链。建议使用独立的现代 Python 环境，不要破坏已经能运行的 PyTorch 2.2 环境：

```bash
cd "/Users/a1523647308/jupyter notebook/pytorch-fluency"
python3.11 -m venv .venv-toolchain
source .venv-toolchain/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-toolchain.in
```

如果本机没有 `python3.11`，使用你环境管理器中可用且受当前依赖支持的 Python 版本。安装完成后把真实版本锁定：

```bash
python -m pip freeze > requirements-toolchain.lock.txt
```

## 2. Shape contract

```text
input_ids:      [B,T] long
attention_mask: [B,T]
labels:         [B] long, values 0/1
logits:         [B,2]
loss:           scalar
```

`T` 由当前 batch 中最长样本决定，因为 `DataCollatorWithPadding` 做动态 padding。

## 3. 先跑自定义循环

默认使用 `distilbert-base-uncased`，第一次运行会下载模型：

```bash
python projects/hf_lora_text_classification/train_custom.py \
  --device auto \
  --epochs 3 \
  --output-dir projects/hf_lora_text_classification/artifacts/lora_custom
```

这个命令默认启用 LoRA。全参数对照：

```bash
python projects/hf_lora_text_classification/train_custom.py \
  --no-lora \
  --epochs 3 \
  --output-dir projects/hf_lora_text_classification/artifacts/full_custom
```

记录：总参数量、可训练参数量、训练时间、adapter/模型目录大小和验证准确率。

## 4. 再用 Trainer 重做

```bash
python projects/hf_lora_text_classification/train_trainer.py \
  --epochs 3 \
  --output-dir projects/hf_lora_text_classification/artifacts/lora_trainer
```

对照 `train_custom.py`，逐项写出 Trainer 隐藏的职责。不要只记 `TrainingArguments` 参数名。

## 5. 重新加载预测

```bash
python projects/hf_lora_text_classification/predict.py \
  --checkpoint projects/hf_lora_text_classification/artifacts/lora_custom/best_model \
  --text "the explanation is clear and useful" \
  --text "the experiment is unreliable and confusing"
```

LoRA checkpoint 只有 adapter，加载时仍需下载或找到原 base model。

## 6. W&B

先执行 `wandb login`，然后：

```bash
python projects/hf_lora_text_classification/train_custom.py \
  --wandb-project pytorch-research-toolchain \
  --wandb-mode online
```

没有提供 `--wandb-project` 时完全不启用 W&B。需要只写本地时使用 `--wandb-mode offline`。

## 7. LoRA target modules

DistilBERT 的 attention 投影层包含 `q_lin`、`k_lin`、`v_lin`、`out_lin`，本项目默认把 LoRA 加到 `q_lin,v_lin`。换模型前先检查：

```python
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)
```

不要照搬 target module 名称；不同架构可能使用 `q_proj/v_proj`、`query/value` 等名称。

项目提供检查脚本：

```bash
python projects/hf_lora_text_classification/inspect_linear_modules.py \
  --model-id distilbert-base-uncased
```

## 8. 刻意练习

1. 打印 tokenization 前后的一个样本；
2. 比较动态 padding 与固定 `max_length` padding 的 token 数；
3. 把标签改成字符串，观察 features 和 collator 如何变化；
4. 比较 LoRA `r=4/8/16`，每次保持其他配置不变；
5. 分别把 target modules 设为 Q/V 与全部 attention linear；
6. 断开网络后用已缓存的 model/dataset 运行；
7. 指定 model revision 并记录到 W&B；
8. 从 `last_model` 恢复 adapter 和 optimizer 继续训练；
9. 给 Trainer 增加早停 callback；
10. 把数据换成你自己的合法文本，保留模块边界。

## 9. 验收

- 能解释每个 batch 字段和 shape；
- 自定义循环与 Trainer 均跑通；
- 能证明 LoRA 只训练少量参数；
- 能保存、重载 adapter 并预测；
- 能指出 target module 错误时该怎样检查；
- W&B run 中包含配置、Git commit 和 train/validation 指标；
- Git 仓库没有提交 token、缓存、原始大数据和 checkpoint。
