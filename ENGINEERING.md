# 科研代码工程化：让实验可读、可复现、可继续

工程化不是把代码拆得越碎越好，而是让一次实验回答四个问题：用了什么配置？训练了什么模型？结果在哪里？能否从中断处继续？

配套项目：[projects/engineered_cnn/README.md](projects/engineered_cnn/README.md)

## 1. 推荐的最小边界

```text
engineered_cnn/
├── config.py       # 参数定义和命令行解析
├── data.py         # 样本生成、Dataset、DataLoader
├── model.py        # 只放模型结构和 shape contract
├── engine.py       # train/evaluate/one-batch check
├── checkpoint.py   # 保存与恢复训练状态
├── utils.py        # seed、device、logging、CSV
├── train.py        # 组装以上模块，不塞具体实现
└── predict.py      # 从最佳权重重新构建模型并推理
```

边界判断方法：如果更换模型时不该影响数据模块，就不要让 `data.py` import 具体模型；如果验证时不该知道日志文件路径，就不要在 `evaluate` 中写文件。

## 2. 配置：禁止让关键数字散落

配置项通常包括：

```text
seed、device、epochs、batch_size、lr、weight_decay
模型宽度、类别数、图像尺寸
数据 worker、输出目录、resume 路径
```

本项目用 argparse 接收命令行、dataclass 在代码中传递：

```bash
python3 projects/engineered_cnn/train.py \
  --epochs 8 \
  --batch-size 64 \
  --lr 0.001 \
  --output-dir projects/engineered_cnn/artifacts/run01
```

运行开始时把最终配置保存成 JSON。以后看结果时，不必猜命令行是什么。

## 3. 可复现：种子只是第一步

最小设置包括 Python、PyTorch 和 CUDA 的随机状态，并关闭会主动寻找非确定性最快算法的设置。即便如此，跨设备、驱动和 PyTorch 版本仍可能有差异。因此结果里还要记录：

- PyTorch 版本；
- device；
- 完整配置；
- 数据划分 seed；
- checkpoint 的 epoch 和指标。

“可复现”通常是统计和流程层面的，不承诺任意硬件上逐 bit 相同。

## 4. logging：训练进度与实验数据分开

标准库 `logging` 同时写终端和 `train.log`，便于人阅读；`metrics.csv` 每轮一行，便于后续画曲线：

```text
epoch,train_loss,train_accuracy,val_loss,val_accuracy
1,0.82,0.71,0.51,0.84
```

本地记录是底座；需要比较大量实验时，再把同一份 metrics 接到 W&B。配套项目提供了可选适配器，默认不产生外部依赖或网络写入。

安装并登录后开启在线记录：

```bash
python3 -m pip install wandb
wandb login
python3 projects/engineered_cnn/train.py \
  --wandb-project pytorch-fluency \
  --epochs 8
```

只在本地记录、稍后再同步：

```bash
python3 projects/engineered_cnn/train.py \
  --wandb-project pytorch-fluency \
  --wandb-mode offline
```

代码只在显式提供 `--wandb-project` 时初始化 W&B；每轮通过 run 对象记录 metrics，结束时显式 `finish()`。具体账号和同步行为以 [W&B 官方 quickstart](https://docs.wandb.ai/models/quickstart) 为准。

## 5. best 和 last 解决不同问题

- `best.pt`：验证指标最好，用于最终评估或推理；
- `last.pt`：最近一次完整训练状态，用于中断恢复。

恢复训练至少要保存 model、optimizer、epoch、best metric。为了更接近连续运行，本项目也保存 CPU/CUDA 和 DataLoader generator 的随机状态。

## 6. 训练入口应当像目录，而不是仓库

理想的 `train.py` 只负责按顺序组装：

```text
parse config
-> create output/logging
-> seed/device
-> build data/model/loss/optimizer
-> optional resume
-> one-batch diagnostic
-> epoch loop
-> metrics/checkpoints
```

如果 `train.py` 里出现几百行模型层定义或数据清洗细节，边界已经泄漏。

## 7. 工程化自测

运行：

```bash
python3 projects/engineered_cnn/train.py --device cpu --epochs 3
python3 projects/engineered_cnn/predict.py \
  --checkpoint projects/engineered_cnn/artifacts/default/best.pt \
  --device cpu
python3 projects/engineered_cnn/train.py \
  --device cpu --epochs 5 \
  --resume projects/engineered_cnn/artifacts/default/last.pt
```

回答：

1. 哪个文件决定输入 shape？
2. 哪个函数能在不修改的情况下换成 MLP？
3. best 与 last 为什么不能只留一个？
4. resume 后从哪个 epoch 开始？
5. 删除 `optimizer_state` 后，“加载权重”和“继续训练”有什么区别？
