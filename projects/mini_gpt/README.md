# 项目二：从零训练一个 MiniGPT

这个项目只依赖 PyTorch，不使用 Hugging Face。目标不是训练有用的大模型，而是把 Transformer 的数据形状、因果注意力、next-token loss、训练循环和生成过程真正串起来。

## 1. 最终数据契约

```text
tokens:  [B, T],       int64
x:       [B, T, D],    token embedding + position embedding
q/k/v:   [B, H, T, Dh]
logits:  [B, T, V]
targets: [B, T],       int64
loss:    CrossEntropy(logits.reshape(B*T,V), targets.reshape(B*T))
```

其中：

- `B`：batch size；
- `T`：上下文长度 block size；
- `D`：embedding dim；
- `H`：attention head 数；
- `Dh = D / H`；
- `V`：字符词表大小。

## 2. next-token 数据为什么要右移

给定文本 `训练循环`：

```text
input:  训 练 循
target: 练 循 环
```

每个位置预测它后面的字符。`NextTokenDataset` 返回长度相同、错开一个 token 的两段。

## 3. embedding

```python
token_x = token_embedding(tokens)       # [B,T] -> [B,T,D]
position_x = position_embedding(arange(T))  # [T] -> [T,D]
x = token_x + position_x
```

token embedding 表示“是什么字符”，position embedding 表示“在第几个位置”。两者最后一维相同，按位置相加。

## 4. causal self-attention

核心计算：

```text
Q,K,V = linear(x)
score = Q @ K^T / sqrt(Dh)
score[future positions] = -inf
attention = softmax(score)
output = attention @ V
```

causal mask 保证位置 `t` 看不到 `t+1` 以后的答案。mask 是下三角矩阵：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

调试注意力时，永远打印 Q/K/V、score、mask 和 output 的形状；不要只盯着最终 loss。

## 5. Decoder block

本项目使用 pre-norm 残差结构：

```python
x = x + attention(layer_norm_1(x))
x = x + mlp(layer_norm_2(x))
```

堆叠若干 block 后，经 LayerNorm 和线性 head 得到 `[B,T,V]` logits。

## 6. 先跑 smoke 配置

从教程根目录执行：

```bash
python3 projects/mini_gpt/train.py \
  --device cpu \
  --max-steps 20 \
  --eval-interval 10 \
  --eval-batches 2 \
  --batch-size 8 \
  --block-size 32 \
  --embed-dim 32 \
  --num-heads 4 \
  --num-layers 1 \
  --output-dir projects/mini_gpt/artifacts/smoke
```

这一步只验证 forward、backward、evaluation、checkpoint 和 generate 全部能走通，不要求生成质量。

## 7. 正式小实验

```bash
python3 projects/mini_gpt/train.py \
  --device auto \
  --max-steps 500 \
  --eval-interval 100 \
  --output-dir projects/mini_gpt/artifacts/run01
```

从最佳 checkpoint 生成：

```bash
python3 projects/mini_gpt/generate.py \
  --checkpoint projects/mini_gpt/artifacts/run01/best.pt \
  --prompt "模型" \
  --max-new-tokens 120 \
  --temperature 0.8
```

继续训练到总计 800 steps：

```bash
python3 projects/mini_gpt/train.py \
  --device auto \
  --max-steps 800 \
  --resume projects/mini_gpt/artifacts/run01/last.pt \
  --output-dir projects/mini_gpt/artifacts/run01
```

## 8. 刻意练习

按顺序完成，每次只改一个变量：

1. 注释 causal mask，比较 validation loss 和生成行为；
2. 把 targets 错误地设成 inputs，解释模型为什么学成“复制器”；
3. 让 `embed_dim` 不能整除 `num_heads`，为错误写清楚断言；
4. 打印每一层的 shape，确认 batch 和 sequence 维没有交换；
5. 对一个 batch 反复训练，确认模型能把它记住；
6. 把 corpus 换成自己的合法文本，重新构建字符词表；
7. 实现 top-k sampling，并比较不同 temperature；
8. 给训练循环增加 gradient accumulation 和 AMP，再做关闭对照。

## 9. 什么时候算掌握

你应当能不看代码回答：

- 为什么 logits 不是概率？
- causal mask 应该遮住上三角还是下三角？
- `D` 如何拆成 `H × Dh`？
- 为什么 Q 乘 K 的转置得到 `[B,H,T,T]`？
- loss 为什么展平 batch 和 time，但保留 vocab 维？
- 推理时为什么只取最后一个位置的 logits？

能从空白实现一个 attention block，并定位上述维度错误，就完成了当前阶段的 Transformer 目标。

