# 项目四：PyTorch DDP 与 DeepSpeed 配置

这个项目把“多卡训练”拆成两层：

1. 用纯 PyTorch DDP 理解进程、数据分片和梯度同步；
2. 用 DeepSpeed JSON 理解训练状态分片与全局 batch。

## 1. DDP shape 与进程契约

```text
world_size: 总进程数
rank:       全局进程编号
local_rank: 当前机器上的进程编号
每进程:    一份完整模型 + 一份不同数据分片
```

DDP 不替你分数据。本项目显式使用 `DistributedSampler`，并在每个 epoch 调用 `set_epoch()`。

## 2. 本机 CPU 两进程演示

这不会加速训练，只用于验证分布式逻辑：

```bash
cd "/Users/a1523647308/jupyter notebook/pytorch-fluency"
torchrun --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29577 \
  projects/distributed_training/ddp_train.py \
  --epochs 3 \
  --output projects/distributed_training/artifacts/ddp_cpu.pt
```

如果当前环境找不到 `torchrun`：

```bash
python -m torch.distributed.run --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29577 \
  projects/distributed_training/ddp_train.py --epochs 3
```

显式使用 `127.0.0.1` 可以避免部分 macOS 环境的主机名/IPv6 解析问题。端口已被占用时换一个未使用端口。

脚本在没有 `torchrun` 时也能单进程运行，便于先调试普通训练。

## 3. 真正的单机多 GPU

在至少两张 CUDA GPU 的机器上使用同一命令。脚本会自动选择 NCCL，并让 `LOCAL_RANK` 对应一张 GPU。

必须检查：

- 每个 rank 处理的样本不同；
- 全局 batch 是否符合预期；
- 只有 rank 0 写日志和 checkpoint；
- loss/accuracy 经过 `all_reduce`；
- checkpoint 使用未包装模型的 state dict；
- 单卡与多卡有效 batch 变化是否需要调整学习率。

## 4. DeepSpeed 配置

项目提供：

- `configs/zero2.json`：optimizer state + gradient 分片；
- `configs/zero3_offload.json`：参数也分片，并 offload 到 CPU。

先在本机验证 JSON 和 batch 关系：

```bash
python projects/distributed_training/validate_deepspeed_configs.py \
  --world-size 2
```

全局 batch 必须满足：

```text
train_batch_size
= train_micro_batch_size_per_gpu
× gradient_accumulation_steps
× world_size
```

## 5. 与 Hugging Face Trainer 连接

安装 DeepSpeed 的 CUDA 环境中：

```bash
python projects/hf_lora_text_classification/train_trainer.py \
  --deepspeed-config projects/distributed_training/configs/zero2.json \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --output-dir projects/hf_lora_text_classification/artifacts/ds_zero2
```

注意 JSON 中的 `train_batch_size` 目前按 `world_size=2` 编写。GPU 数、micro batch 或累积步数改变后必须同步修改，或者让集成层使用 `auto` 配置。

## 6. 什么时候用什么

| 场景 | 工具 |
|---|---|
| 模型能放单卡，只想提高吞吐 | DDP |
| optimizer/gradient 让显存不够 | ZeRO-1/2 |
| 模型参数本身放不进单卡 | ZeRO-3/FSDP |
| GPU 仍不够，需要使用主机内存 | ZeRO offload |

offload 是容量手段，不保证更快。通信、CPU 和存储带宽可能成为新瓶颈。

## 7. 故障练习

1. 注释 `sampler.set_epoch()`，比较各 epoch 数据顺序；
2. 删除 `all_reduce`，观察 rank 0 指标为何不是全局指标；
3. 让所有 rank 同时保存同一路径，解释竞态风险；
4. 修改 world size，不修改 DeepSpeed `train_batch_size`；
5. 故意让一个 rank 抛异常，观察其他 rank 的表现；
6. 把 per-device batch 减半、accumulation 加倍，验证 global batch 不变。

## 8. 验收

- 能画出 rank/local rank/world size；
- 能解释 sampler 和梯度同步分别解决什么；
- CPU 两进程脚本能运行；
- 只有一个 checkpoint；
- 能计算 global batch；
- 能解释 ZeRO-1/2/3 分片差异；
- 知道分片 checkpoint 需要专用恢复/合并流程。
