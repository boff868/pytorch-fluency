# 第三层：LLM / ML 研究工具链

这一层不再主要学习“怎样写一个 Linear 层”，而是学习如何把预训练模型、真实数据、实验记录和多 GPU 资源接成可复现的研究流程。

推荐顺序：

```text
Transformers + Datasets
-> 自定义 PyTorch 微调循环
-> Trainer
-> PEFT / LoRA
-> W&B + Git 实验管理
-> DDP
-> DeepSpeed ZeRO
```

DDP 和 DeepSpeed 不应挡在前面：模型还没在单卡正确训练时，不要先增加多进程和分片复杂度。

配套代码：

- [Hugging Face + LoRA 文本分类项目](projects/hf_lora_text_classification/README.md)
- [PyTorch DDP 项目](projects/distributed_training/README.md)
- [Git 与实验版本管理](GIT_EXPERIMENTS.md)

---

## 1. Transformers：把预训练模型接入自己的任务

### 要理解的对象

```text
model id / revision
-> tokenizer
-> input_ids + attention_mask
-> AutoModelFor<Task>
-> ModelOutput(loss, logits, ...)
-> save_pretrained / from_pretrained
```

必须掌握：

- `AutoTokenizer.from_pretrained()`；
- `AutoModelForSequenceClassification` 与 `AutoModelForCausalLM`；
- padding、truncation、`max_length`；
- `input_ids`、`attention_mask`、`labels` 的 shape；
- `model(**batch)` 返回的 `loss` 和 `logits`；
- `save_pretrained()`、重新加载和推理；
- 生成任务中的 `generate()`、temperature、top-k；
- 模型 revision、下载缓存和离线使用。

不要只学 `pipeline()`。它适合快速推理，却隐藏了 batch、mask、loss 和 device，不能替代训练能力。

### 两种训练方式

先用自定义 PyTorch 循环：

```python
outputs = model(**batch)
loss = outputs.loss
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

再用 `TrainingArguments + Trainer` 重做同一任务。你必须能指出 Trainer 替你完成了哪些事情：DataLoader、梯度更新、评估、checkpoint、混合精度、日志和分布式启动。

验收：同一模型和数据分别用自定义循环、Trainer 跑通；能够解释指标不完全一致可能来自哪些随机性和默认参数。

---

## 2. Datasets：让数据处理可缓存、可复用、可检查

需要掌握：

- `load_dataset()` 加载 Hub、本地 CSV/JSON/Parquet；
- `Dataset`、`DatasetDict`、train/validation/test split；
- `map(batched=True)` 批量 tokenize；
- `filter()`、`select()`、`shuffle()`、`train_test_split()`；
- `remove_columns`、`rename_column`；
- features、label 类型和类别映射；
- fingerprint/cache；
- streaming dataset 的限制；
- data collator 的动态 padding。

数据契约示例：

```text
原始样本：{"text": str, "label": int}
tokenized：{"input_ids": list[int], "attention_mask": list[int], "label": int}
collated batch：
  input_ids      [B,T] long
  attention_mask [B,T] long/bool
  labels         [B]   long
```

重点不是 `map()` 的语法，而是保证：训练/验证没有泄漏、标签映射一致、无用列不进入模型、padding 不浪费过多计算。

验收：能从本地 CSV 构造 DatasetDict，批量 tokenize，动态 padding，并打印一个完全符合模型契约的 batch。

---

## 3. PEFT / LoRA：只训练小量增量参数

LoRA 保持原权重 `W` 冻结，学习低秩增量：

```text
W' = W + scaling × B × A
```

需要掌握：

- base model 与 adapter 的关系；
- `LoraConfig`、`get_peft_model()`、`PeftModel`；
- `r`、`lora_alpha`、`lora_dropout`；
- `target_modules` 为什么依赖模型结构；
- `TaskType`；
- 查看 trainable parameters；
- adapter 的保存、加载与合并；
- LoRA、量化与 QLoRA 的区别。

必须做一次对照实验：

| 对照项 | 全参数微调 | LoRA |
|---|---:|---:|
| 总参数量 | 记录 | 记录 |
| 可训练参数量 | 记录 | 记录 |
| 训练耗时 | 记录 | 记录 |
| 峰值内存 | 记录 | 记录 |
| checkpoint 大小 | 记录 | 记录 |
| validation 指标 | 记录 | 记录 |

验收：能解释为什么只保存 adapter 仍需要 base model；能根据模型的 `named_modules()` 找到正确的 LoRA target modules。

---

## 4. W&B：让每条曲线对应一个真实实验

当前工程化 CNN 已覆盖 `wandb.init()`、config、`run.log()`、online/offline 和 `finish()`。第三层继续掌握：

- project、run、group、tag；
- 明确使用 epoch 或 global step 作为横轴；
- summary/best metric；
- checkpoint 与 dataset artifacts；
- resume 同一个 run；
- sweep 做超参数搜索；
- 表格、混淆矩阵和样本预测；
- Git commit 与 run 的关联。

最低记录集合：

```text
train/val loss
主要任务指标
learning rate
gradient norm
吞吐量与显存
best epoch
完整配置
Git commit
数据/model revision
```

验收：至少完成四个可比较 run，一次只改变一个因素，并能从 dashboard 说清哪条曲线对应哪次代码和配置。

---

## 5. Git/GitHub：让实验结果对应确定代码

需要掌握：

- `status`、`diff`、`add`、`commit`；
- branch、merge/rebase 的用途；
- pull request 和 code review；
- tag/release；
- `.gitignore`；
- requirements/环境锁定；
- commit hash；
- 从历史版本复现实验。

科研实验流程：

```text
检查 diff
-> 提交本次实验代码
-> 获取 commit hash
-> 启动 W&B run，记录 commit/config
-> 保存指标和 artifact
-> 在 README/实验表写结论
```

不要提交 API key、`.env`、虚拟环境、下载缓存、原始大数据和普通 checkpoint。需要 Git 跟踪的大文件使用 Git LFS；大量生成文件更适合对象存储、模型 Hub 或 W&B artifact。

详细命令见 [GIT_EXPERIMENTS.md](GIT_EXPERIMENTS.md)。

---

## 6. DDP：模型每张卡一份，数据各算一部分

DDP 的基本结构：

```text
一个进程 <-> 一张 GPU <-> 一份完整模型 <-> 一份数据分片
                             |
                         梯度 all-reduce
```

必须理解：

- rank、local rank、world size；
- process group 与 NCCL/Gloo；
- `torchrun`；
- `DistributedDataParallel`；
- `DistributedSampler` 与 `sampler.set_epoch(epoch)`；
- 只让 rank 0 打日志和保存；
- 使用 `all_reduce` 聚合全局指标；
- `model.module.state_dict()`；
- global batch size。

```text
global batch
= per-device batch
× world size
× gradient accumulation steps
```

典型错误：所有进程读取同一批数据、每个进程重复写 checkpoint、只统计 rank 0 指标、某个进程异常后其他进程永久等待、global batch 改变却没有重新考虑学习率。

配套项目可以在 CPU/Gloo 上验证进程逻辑；真实加速需要多 GPU 环境。

验收：两进程启动成功、数据分片不同、全局指标聚合正确、只有 rank 0 生成 checkpoint。

---

## 7. DeepSpeed：模型放不进单卡时再深入

DDP 在每张 GPU 都保留完整模型。DeepSpeed ZeRO 则逐步分片训练状态：

| ZeRO 阶段 | 分片内容 |
|---|---|
| 1 | optimizer states |
| 2 | optimizer states + gradients |
| 3 | optimizer states + gradients + parameters |

需要理解：

- micro batch、gradient accumulation、global batch；
- fp16/bf16；
- ZeRO-1/2/3 的显存与通信权衡；
- optimizer/parameter CPU offload；
- DeepSpeed JSON；
- 分片 checkpoint、resume 与 fp32 权重合并；
- Hugging Face Trainer 的 `deepspeed` 配置入口。

不要把 ZeRO-3 当作默认配置：stage 越高，通信和保存恢复越复杂。先确认模型是否真的放不进单卡，以及瓶颈是参数、梯度、optimizer state 还是 activation。

当前机器未安装 DeepSpeed，也没有多 CUDA GPU。教程提供可审查的 JSON 配置和批大小验证脚本；真正训练安排到 NVIDIA GPU 服务器。

验收：能根据模型是否放得进单卡选择 DDP/ZeRO-2/ZeRO-3；能算清 DeepSpeed 配置里的 global batch；能解释分片 checkpoint 为什么不能直接当普通 `state_dict`。

---

## 8. 推荐的四周工具链安排

### 第 7 周：Transformers + Datasets

- 加载 tokenizer 和预训练分类模型；
- 处理本地 CSV；
- 先自定义循环，后 Trainer；
- 保存、重载、预测。

### 第 8 周：PEFT / LoRA

- 找 target modules；
- 做全参数/LoRA 对照；
- 保存 adapter；
- 记录可训练参数和 checkpoint 大小。

### 第 9 周：W&B + Git/GitHub

- 运行至少四次单变量实验；
- 每个 run 记录 commit/config；
- 上传最佳 adapter artifact；
- 使用 branch/PR 整理实验代码。

### 第 10 周：DDP + DeepSpeed

- 本机 CPU 跑两进程 DDP 教学实验；
- 阅读 rank/sampler/all-reduce；
- 验证两个 DeepSpeed JSON 的 batch 关系；
- 有多 GPU 资源时再做真实训练。

完成第 9 周已经足够进行常规单卡 LLM/ML 微调研究；第 10 周是按项目规模持续推进的能力。

---

## 9. 官方资料

- [Transformers 微调](https://huggingface.co/docs/transformers/training)
- [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Datasets 加载](https://huggingface.co/docs/datasets/loading)
- [PEFT](https://huggingface.co/docs/peft/en/index)
- [LoRA](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
- [PyTorch DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [DeepSpeed 入门](https://www.deepspeed.ai/getting-started/)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [W&B Quickstart](https://docs.wandb.ai/models/quickstart)

