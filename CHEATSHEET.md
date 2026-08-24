# PyTorch 独立编程速查

## 1. 先写契约

```text
x:      [B, ...], dtype=?, device=?
logits: [B, C]
y:      [B], long, range=[0, C-1]
loss:   scalar
```

## 2. 网络

```python
class Model(nn.Module):
    def __init__(self, in_features, hidden, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)
```

## 3. 训练一个 batch

```python
model.train()
x, y = x.to(device), y.to(device)
logits = model(x)
loss = loss_fn(logits, y)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

口诀：**前损清反更**——前向、损失、清梯度、反传、更新。

## 4. 验证

```python
model.eval()
with torch.inference_mode():
    logits = model(x)
```

`train/eval` 控制层行为；`inference_mode` 控制是否记录梯度。两者不是一回事。

## 5. 正确聚合指标

```python
n = y.shape[0]
loss_sum += loss.item() * n
correct += (logits.argmax(1) == y).sum().item()
sample_count += n

mean_loss = loss_sum / sample_count
accuracy = correct / sample_count
```

## 6. 多分类约定

```text
model output: [B, C] float logits；不要先 softmax
target:       [B] long，值域 0..C-1
loss:         nn.CrossEntropyLoss()
```

## 7. 调试顺序

```text
CPU + 一个 batch
-> shape / dtype / device / finite
-> dummy forward
-> 过拟合一个 batch
-> 梯度是否存在
-> 参数是否更新
-> train/eval 是否正确
-> 学习率与数据
-> 最后才优化性能
```

## 8. 核心断言

```python
assert logits.ndim == 2
assert y.ndim == 1
assert logits.shape[0] == y.shape[0]
assert y.dtype == torch.long
assert logits.device == y.device
assert torch.isfinite(logits).all()
assert y.min() >= 0
assert y.max() < logits.shape[1]
```

## 9. checkpoint

```python
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
}, path)

ckpt = torch.load(path, map_location=device)
model.load_state_dict(ckpt["model_state"])
optimizer.load_state_dict(ckpt["optimizer_state"])
```

## 10. 卡住时问自己

1. 每个维度分别代表什么？
2. loss 想要什么 shape 和 dtype？
3. 哪一步开始出现 NaN/Inf？
4. 每个参数都有合理梯度吗？
5. 单 batch 能被记住吗？
6. 验证时同时用了 `eval()` 和无梯度上下文吗？

## 11. 科研项目最小结构

```text
config -> data -> model -> engine -> metrics/checkpoint
```

每次实验至少保留：最终配置、软件/device 信息、日志、逐轮指标、`best.pt` 和 `last.pt`。

## 12. Transformer 形状

```text
tokens  [B,T]
embed   [B,T,D]
q/k/v   [B,H,T,Dh]，D = H * Dh
scores  [B,H,T,T]
logits  [B,T,V]
target  [B,T]
```

语言模型交叉熵：把 logits 变成 `[B*T,V]`，targets 变成 `[B*T]`。生成时只使用最后一个位置的 `[B,V]` logits。

## 13. Hugging Face 分类 batch

```text
input_ids       [B,T] long
attention_mask  [B,T]
labels          [B] long
logits          [B,C]
```

先用 `model(**batch)` 接回你自己的训练循环，再学习 Trainer。动态 padding 由 data collator 在组 batch 时完成。

## 14. LoRA

```text
W' = W + scaling * B * A
```

检查：base 是否冻结、target modules 是否匹配、可训练参数比例、adapter 是否能与 base model 重新组合。

## 15. DDP 与 DeepSpeed

```text
global_batch = per_device_batch * world_size * accumulation_steps
```

- DDP：每张 GPU 完整模型，数据分片，梯度同步；
- ZeRO-1：分 optimizer states；
- ZeRO-2：再分 gradients；
- ZeRO-3：再分 parameters。

DDP 必查：rank、sampler、`set_epoch`、all-reduce、rank-zero checkpoint。
