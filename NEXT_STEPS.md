# 接下来 10 周：从“会写循环”到“能独立做研究实验”

这份路线接在主教程之后。目标不是增加阅读量，而是逐步撤掉参考代码。每天建议 45～90 分钟，每周至少留一天只做复盘和默写。

## 使用规则

1. 每个项目先读任务，不先读参考实现；
2. 从空白文件写 20～30 分钟，卡住后只查某个 API，不复制整段代码；
3. 跑通后再与参考代码逐行对照；
4. 每次 bug 记录“症状、最小复现、根因、新增断言”；
5. 下一周开始前，重新默写上一周最核心的函数。

## 第 1 周：把基础循环写进肌肉记忆

材料：主教程第 1～6 章、`labs/01`～`labs/04`。

任务：

- 默写 MLP、`train_one_epoch`、`evaluate`，各 3 次；
- 分别制造 shape、dtype、广播和无梯度错误；
- 给训练循环加单 batch 过拟合检查；
- 不看答案完成 `exercises/README.md` 的练习 1～5。

周末验收：20 分钟内从空白写出多分类训练循环；能解释 `train/eval` 和 grad mode 是两套开关。

## 第 2 周：完成第一个独立项目

材料：`labs/03_train_classifier.py`，但开始写代码后不要打开它。

任务：

- 自己生成三分类二维数据；
- 独立划分 train/validation；
- 写 MLP、训练、验证、最佳 checkpoint；
- 画图不是必需，先保证指标和恢复正确；
- 完成毕业题并按 100 分评分表自评。

周末验收：新模型加载 checkpoint 后，同一输入的 logits 与保存状态一致。

## 第 3 周：CNN 与工程化

材料：[科研代码工程化](ENGINEERING.md) 和 `projects/engineered_cnn/`。

第一遍：只读项目 README 的任务，自己拆成：

```text
config -> data -> model -> engine -> checkpoint/logging -> train
```

第二遍：运行参考项目并逐个删除一个关键调用，观察测试怎样失败。重点掌握：

- argparse + dataclass 配置；
- 随机种子与确定性设置；
- 标准 logging、CSV 指标；
- 数据、模型、循环、工具分模块；
- `last.pt` 继续训练与 `best.pt` 模型选择。

周末验收：训练循环不修改，把 MLP 数据/模型换成 CNN 数据/模型；能从 `last.pt` 接着跑。

## 第 4 周：Transformer 最小原理

材料：`projects/mini_gpt/README.md` 的第 1～5 节。

任务：

- 手写字符 tokenizer；
- 写 token embedding 与 position embedding；
- 用一个小矩阵手算 causal mask；
- 手写多头自注意力的 Q、K、V reshape；
- 逐层断言 `[B,T] -> [B,T,D] -> [B,H,T,Dh] -> [B,T,V]`。

周末验收：不用看实现，能写出 attention 的维度变化和 causal mask；能说明为什么语言模型 loss 要把 `[B,T,V]` 与 `[B,T]` 展平。

## 第 5 周：训练 MiniGPT

材料：`projects/mini_gpt/` 完整项目。

任务：

- 先运行 smoke 配置，证明 forward/backward/generate 通；
- 训练 bundled corpus，观察 train/validation loss；
- 从 checkpoint 生成文本；
- 分别改变 block size、embedding dim、head 数，一次只改一个变量；
- 故意破坏 mask、目标右移和 head 维度，记录错误表现。

周末验收：独立写一个 Decoder block；能从 logits 采样下一个 token，并循环生成。

## 第 6 周：脱离教程完成综合项目

从下面任选一个：

- 把二维 MLP 项目换成真实 CSV 分类数据；
- 把 CNN 合成数据换成 FashionMNIST/CIFAR-10；
- 把 MiniGPT 语料换成自己的合法文本；
- 给已有研究代码重构配置、日志、checkpoint 和模块边界。

必须交付：

```text
README：任务、数据、运行命令、结果
config：所有重要超参数
data：数据契约与划分
model：shape contract
engine：训练/验证
artifacts：日志、指标、checkpoint（大权重可忽略提交）
debug-log：至少 3 个真实 bug 的证据链
```

## 第 7 周：Transformers + Datasets

材料：`RESEARCH_TOOLCHAIN.md` 第 1～2 节、`projects/hf_lora_text_classification/`。

- 从本地 CSV 创建 DatasetDict；
- 批量 tokenize 和动态 padding；
- 用预训练分类模型配合自定义 PyTorch 循环；
- 保存、重新加载并预测；
- 逐个解释 `input_ids/attention_mask/labels/logits`。

周末验收：不用 Trainer 也能微调预训练模型，并能证明 batch 契约正确。

## 第 8 周：Trainer + PEFT / LoRA

- 用 Trainer 重做第 7 周任务；
- 对照 Trainer 隐藏的训练步骤；
- 找到模型的 Linear 模块名称；
- 完成全参数与 LoRA 对照；
- 比较可训练参数、耗时、内存、权重大小和验证指标。

周末验收：能够保存 adapter，在新进程中重新组合 base model + adapter 推理。

## 第 9 周：W&B + Git/GitHub

材料：`GIT_EXPERIMENTS.md`。

- 至少进行四次单变量实验；
- 每个 W&B run 记录完整配置和 Git commit；
- 记录 best metric 和 adapter artifact；
- 使用实验分支和 pull request 写结论；
- 检查 `.gitignore` 没有泄漏 token、缓存和权重。

周末验收：随机选择一条曲线，可以定位到确定 commit、配置、数据/model revision 和输出 artifact。

## 第 10 周：DDP + DeepSpeed

- 本机 CPU/Gloo 跑两进程 DDP；
- 理解 rank/local rank/world size；
- 检查 DistributedSampler、all-reduce 和 rank-zero 保存；
- 验证 DeepSpeed JSON 的 global batch；
- 解释 ZeRO-1/2/3；
- 有多 GPU 资源时再做真实加速实验。

周末验收：能判断一个任务应该继续单卡、使用 DDP，还是需要 ZeRO 分片。

## PyTorch 基础毕业标准

满足 8 条中的 7 条，就达到了“独立做课程项目和常规科研实验”的 PyTorch 熟练度：

- 30 分钟内从空白写出完整监督学习流程；
- 不靠 trial-and-error 猜形状，先写 contract；
- 单 batch 过拟合测试会主动使用；
- 能定位 shape、dtype、device、NaN、无梯度问题；
- 能正确聚合训练/验证指标；
- 能保存 best、恢复 last 并继续训练；
- 能把单文件训练重构成清楚的模块；
- 能实现并训练一个最小 causal Transformer。

达到这里就先去做真实任务，不必继续堆 API。DDP、自定义 CUDA、复杂编译优化等内容等真正遇到规模瓶颈再学。

## 研究工具链毕业标准

- 自定义循环与 Trainer 都能微调预训练模型；
- 能从本地/Hub 数据构建 tokenized Dataset；
- 能完成 LoRA 全流程并重载 adapter；
- W&B 实验对应确定 Git commit；
- DDP 两进程的数据、指标和保存逻辑正确；
- 能解释 ZeRO 阶段与 global batch，但不要求当前机器真实训练大模型。
