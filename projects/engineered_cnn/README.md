# 项目一：工程化 CNN 分类器

这个项目用纯 PyTorch 生成三类 16×16 灰度图：竖线、横线、对角线。数据是合成的，所以不需要联网；训练管线却与真实图像分类完全一致。

## 能力目标

- 把单文件脚本拆成清楚模块；
- 使用 argparse + dataclass 管理配置；
- 记录日志、CSV 指标和完整 JSON 配置；
- 可选把相同指标记录到 W&B；
- 固定随机种子；
- 保存 `best.pt` 与 `last.pt`；
- 中断后恢复 optimizer 和随机状态；
- 用同一个 engine 训练不同 `nn.Module`。

## Shape contract

```text
x:      [B, 1, 16, 16], float32
logits: [B, 3],           float32
y:      [B],              int64, values 0..2
loss:   scalar
```

## 先做再看

在阅读 `.py` 文件前，自己画模块依赖，并写出每个文件只允许承担的职责。然后从空白实现 `model.py` 和 `engine.py`。

## 运行

从教程根目录执行：

```bash
python3 projects/engineered_cnn/train.py --device cpu --epochs 8
python3 projects/engineered_cnn/predict.py \
  --checkpoint projects/engineered_cnn/artifacts/default/best.pt \
  --device cpu
```

恢复训练：

```bash
python3 projects/engineered_cnn/train.py \
  --device cpu \
  --epochs 12 \
  --resume projects/engineered_cnn/artifacts/default/last.pt
```

`--epochs` 表示最终要达到的总 epoch 数，不是“额外再跑多少轮”。

可选 W&B（先安装 `wandb` 并完成登录）：

```bash
python3 projects/engineered_cnn/train.py \
  --wandb-project pytorch-fluency \
  --wandb-mode online
```

## 输出

```text
artifacts/default/
├── config.json
├── train.log
├── metrics.csv
├── best.pt
└── last.pt
```

## 改造任务

1. 把 `SmallCNN` 换成 MLP：只允许改 `model.py` 和输入 flatten 位置；
2. 加第四类“圆环”，更新 shape contract 和类别数；
3. 加 `StepLR`，把 scheduler state 一并保存；
4. 让 `predict.py` 输出混淆矩阵；
5. 把合成数据换成 torchvision 数据集，保持 `engine.py` 不动。
