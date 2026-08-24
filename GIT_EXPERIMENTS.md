# Git/GitHub 与实验版本管理

目标：任何一条实验曲线都能回答“哪份代码、哪个配置、哪份数据、哪个模型 revision 产生了它”。

## 1. 实验前检查

```bash
git status --short
git diff
git diff --staged
```

只提交与本次实验相关的改动：

```bash
git add path/to/relevant_file.py config.json
git commit -m "experiment: compare LoRA rank 8"
git rev-parse HEAD
```

最后一条输出的 commit hash 应进入实验日志或 W&B config。

## 2. 分支建议

```text
main                 可复现的稳定代码
experiment/lora-rank LoRA 对照实验
fix/data-leakage     数据泄漏修复
```

一个分支解决一个研究问题。完成后通过 pull request 写清：

- 假设是什么；
- 改了哪些变量；
- 数据和评估协议是否改变；
- 结果如何；
- 是否应合入主线。

## 3. `.gitignore` 最小内容

```gitignore
.env
.venv/
__pycache__/
*.pyc
wandb/
artifacts/
checkpoints/
data/raw/
*.pt
*.pth
*.safetensors
```

如果某个小型权重是项目不可缺少的正式资源，再明确决定放 Hub、release、Git LFS 或 artifact store，不要因为忘写 ignore 意外提交。

## 4. 环境记录

依赖清单分两层：

```text
requirements-toolchain.in   人工维护的直接依赖
requirements.lock.txt       实验环境解析出的完整版本
```

环境确认后再生成锁定文件：

```bash
python -m pip freeze > requirements.lock.txt
python -c "import torch; print(torch.__version__)"
```

同时记录 Python、CUDA、GPU、driver、模型 revision 和数据 revision。

## 5. 一次实验的最小记录

```json
{
  "question": "LoRA rank 8 是否优于 rank 4？",
  "git_commit": "...",
  "seed": 17,
  "model_id": "...",
  "model_revision": "...",
  "dataset_revision": "...",
  "changed_variable": "lora_r",
  "baseline_run": "...",
  "result": "...",
  "conclusion": "..."
}
```

一次只改变一个主要因素。若同时改变数据划分、模型和学习率，即使结果变好，也很难确定原因。

## 6. 大文件边界

- 源代码、配置、短报告：Git；
- 小型 adapter：Hugging Face Hub、W&B artifact 或 release；
- 大模型权重：模型 Hub/对象存储；
- 大型原始数据：数据平台/对象存储；
- 必须由 Git 管理的大二进制：Git LFS；
- API token：环境变量或 secret manager，绝不提交。

## 7. 验收任务

1. 建立一个实验分支；
2. 提交基线代码并记录 commit hash；
3. 跑 LoRA rank 4 与 rank 8 两次实验；
4. 在 W&B 中关联各自配置与 commit；
5. 把最佳 adapter 作为 artifact；
6. 写一份包含假设、结果、局限的 pull request 描述；
7. 重新 clone 到空目录，仅凭 README 和锁定依赖恢复推理。

