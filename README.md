# PyTorch 独立实战教程

目标不是“看懂一份训练模板”，而是让你面对一个新任务时，能从空白 `.py` 文件独立完成：

1. 判断输入、输出和标签的形状；
2. 用 `nn.Module` 搭出网络；
3. 写出正确的训练与验证循环；
4. 保存、加载并继续训练；
5. 用固定顺序定位 shape、dtype、device、梯度和数值问题。

## 从这里开始

- 主教程：[TUTORIAL.md](TUTORIAL.md)
- 一页速查：[CHEATSHEET.md](CHEATSHEET.md)
- 无代码骨架的默写题：[exercises/README.md](exercises/README.md)
- 可运行实验：[labs/](labs/)
- 接下来 10 周：[NEXT_STEPS.md](NEXT_STEPS.md)
- 科研代码工程化：[ENGINEERING.md](ENGINEERING.md)
- 第三层研究工具链：[RESEARCH_TOOLCHAIN.md](RESEARCH_TOOLCHAIN.md)
- Git 与实验版本管理：[GIT_EXPERIMENTS.md](GIT_EXPERIMENTS.md)
- 模块化 CNN 项目：[projects/engineered_cnn/README.md](projects/engineered_cnn/README.md)
- MiniGPT / Transformer 项目：[projects/mini_gpt/README.md](projects/mini_gpt/README.md)
- Hugging Face + LoRA 项目：[projects/hf_lora_text_classification/README.md](projects/hf_lora_text_classification/README.md)
- DDP + DeepSpeed 项目：[projects/distributed_training/README.md](projects/distributed_training/README.md)

建议不要一口气读完。每学完一节，关掉教程，在新文件里默写一次；卡住 10 分钟后才能查看答案。

## 运行方式

本教程只依赖 PyTorch，核心实验不下载数据集。

```bash
cd "/Users/a1523647308/jupyter notebook/pytorch-fluency"
python3 -c "import torch; print(torch.__version__)"
python3 labs/01_tensor_autograd.py
python3 labs/02_build_network.py
python3 labs/03_train_classifier.py --device cpu --epochs 8
python3 labs/04_debug_lab.py
python3 labs/05_cnn_shapes.py
python3 smoke_check.py
python3 research_toolchain_check.py
```

如果机器有 CUDA 或 Apple Silicon，可把 `--device cpu` 改为 `cuda` 或 `mps`。第一次排错时仍建议先在 CPU 上跑最小样本。

## 推荐训练节奏

| 阶段 | 内容 | 达标动作 |
|---|---|---|
| 1 | 张量、形状、autograd | 能手算并验证一个线性层的梯度形状 |
| 2 | `nn.Module` | 能根据 shape contract 独立写 MLP/CNN |
| 3 | Dataset/DataLoader | 能说清一个 batch 的每个维度 |
| 4 | 训练/验证循环 | 15 分钟内从空白写出并跑通 |
| 5 | 调试 | 能让模型过拟合单个 batch，再逐层扩大问题 |
| 6 | 综合项目 | 不看答案完成 `exercises/README.md` 的毕业题 |

完成标准不是“所有代码都记得”，而是你能说出每一步为什么存在，并在忘记具体 API 时知道该查哪个对象。

## 完整学习顺序

不要直接跳到 Transformer。按下面顺序，前一阶段验收通过再进入下一阶段：

| 阶段 | 材料 | 产出 |
|---|---|---|
| A. 核心基础 | `TUTORIAL.md` 第 0～7 章、`labs/01`～`05` | 从空白写 MLP/CNN 和训练循环 |
| B. 独立项目 | `exercises/README.md` 毕业题 | 三分类项目、验证和 checkpoint |
| C. 科研工程化 | `ENGINEERING.md`、`projects/engineered_cnn/` | 配置、日志、模块化、恢复训练 |
| D. Transformer | `projects/mini_gpt/README.md` | 手写 attention、训练并生成文本 |
| E. 脱离教程 | `NEXT_STEPS.md` 第 6 周 | 自选真实数据综合项目 |
| F. 研究工具链 | `RESEARCH_TOOLCHAIN.md`、HF/LoRA 项目 | 预训练模型、Datasets、LoRA、W&B/Git |
| G. 分布式概念 | DDP/DeepSpeed 项目 | rank、sampler、ZeRO 与全局 batch |

如果时间有限，优先完成 A～C；做 LLM/现代 ML 研究时继续完成 D～F。G 在有多 GPU 资源或模型容量压力时深入。
