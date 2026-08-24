# PyTorch：从空白文件到独立训练与调试

## 0. 先建立正确的脑内地图

绝大多数 PyTorch 监督学习任务都能拆成六个对象：

```text
Dataset -> DataLoader -> batch
                         |
                         v
输入 x -> model -> 原始输出 logits -> loss(logits, y)
             ^                    |
             |                    v
             +---- optimizer <- gradients
```

对应到代码就是：

```python
dataset = ...
loader = DataLoader(dataset, batch_size=...)
model = MyModel(...).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=...)
```

训练的一个 batch 只有五个必需动作：

```python
logits = model(x)                       # 1. 前向
loss = loss_fn(logits, y)              # 2. 标量损失
optimizer.zero_grad(set_to_none=True)  # 3. 清旧梯度
loss.backward()                        # 4. 算新梯度
optimizer.step()                       # 5. 更新参数
```

先把这张图和这五步记住。数据集、网络类型、损失函数和指标会变，但骨架不变。

---

## 1. 张量：先问 shape，再问数值

### 1.1 四个永远要检查的属性

```python
print(x.shape)   # 每个维度代表什么？
print(x.dtype)   # float32、int64，还是别的？
print(x.device)  # cpu、cuda、mps？
print(torch.isfinite(x).all())  # 有没有 NaN / Inf？
```

写网络之前先写 **shape contract（形状契约）**。例如一个 10 类图片分类器：

```text
x:      [B, 1, 28, 28], float32
logits: [B, 10],         float32
y:      [B],             int64，取值 0..9
loss:   []，              标量
```

`B` 是 batch size。它通常应当是网络定义里“不需要知道具体数值”的维度。

### 1.2 高频形状操作

```python
x = torch.randn(32, 3, 28, 28)

x_flat = x.flatten(start_dim=1)  # [32, 2352]，保留 batch 维
x_one = x.unsqueeze(1)           # [32, 1, 3, 28, 28]
x_back = x_one.squeeze(1)        # [32, 3, 28, 28]
x_hwc = x.permute(0, 2, 3, 1)   # [32, 28, 28, 3]
```

不要把 `view` 当成“随便改形状”。元素总数必须一致，而且某些 `permute` 后的张量内存不连续。日常优先用语义更清楚的 `flatten`、`unsqueeze`、`permute`；确实需要重排时用 `reshape`。

### 1.3 广播：方便，但也会静悄悄地制造 bug

```python
prediction = torch.randn(32, 1)
target = torch.randn(32)
bad = (prediction - target)  # 结果是 [32, 32]，通常不是你想要的！
```

因为 `[32, 1]` 和 `[32]` 会广播成 `[32, 32]`。回归任务中，预测和标签进入 loss 前最好直接断言：

```python
assert prediction.shape == target.shape
```

### 1.4 autograd 到底做了什么

只要参与计算的叶子张量需要梯度，PyTorch 就会在前向计算时记录计算图：

```python
w = torch.tensor([2.0], requires_grad=True)
x = torch.tensor([3.0])
loss = (w * x - 1.0).pow(2).mean()
loss.backward()
print(w.grad)
```

梯度默认**累加**到 `.grad`，所以每个训练 batch 都要清梯度。`loss.item()` 会把单元素张量取成 Python 数字，适合日志；不要在构造 loss 的中间路径上随意 `.item()`，否则会切断计算图。

动手运行：

```bash
python3 labs/01_tensor_autograd.py
```

然后不看文件回答：为什么第二次 `backward()` 前如果不清梯度，结果会变成第一次的两倍？

---

## 2. 搭网络：`__init__` 注册零件，`forward` 描述数据流

### 2.1 最小的可训练网络

```python
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)
```

三个关键点：

1. 必须调用 `super().__init__()`；
2. 可训练层要成为 `self.xxx` 的属性，PyTorch 才能注册参数；
3. 调用 `model(x)`，不要手动调用 `model.forward(x)`，因为前者会保留 hooks 等框架行为。

检查参数：

```python
model = MLP(20, 64, 4)
for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.requires_grad)
```

### 2.2 为什么分类器最后通常不写 Softmax

多分类训练常用：

```python
loss_fn = nn.CrossEntropyLoss()
logits = model(x)          # [B, C]，原始分数
loss = loss_fn(logits, y)  # y: [B]、torch.long
```

`CrossEntropyLoss` 直接接收 logits。训练前在模型尾部再加 `Softmax`，会造成重复归一化并损害数值稳定性。只有展示概率时才做：

```python
probabilities = logits.softmax(dim=1)
predictions = logits.argmax(dim=1)
```

常见任务约定：

| 任务 | 模型输出 | 标签 | 损失 |
|---|---|---|---|
| 单标签多分类 | `[B, C]` logits | `[B]` `long` | `CrossEntropyLoss` |
| 二分类/多标签 | `[B]` 或 `[B, C]` logits | 同形状 `float` | `BCEWithLogitsLoss` |
| 回归 | 与目标同形状 | `float` | `MSELoss` / `L1Loss` |

### 2.3 不要心算复杂卷积尺寸：让网络自己证明

卷积的常见 shape 是 `[B, C, H, W]`。先拿假数据跑一次：

```python
dummy = torch.randn(2, 3, 64, 64)
with torch.inference_mode():
    output = model(dummy)
assert output.shape == (2, 10)
```

减少全连接层输入尺寸的心算，可以用 `nn.AdaptiveAvgPool2d((1, 1))`：

```python
self.features = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
)
self.classifier = nn.Linear(64, 10)

def forward(self, x):
    x = self.features(x)       # [B, 64, 1, 1]
    x = x.flatten(start_dim=1) # [B, 64]
    return self.classifier(x)  # [B, 10]
```

动手运行：

```bash
python3 labs/02_build_network.py
python3 labs/05_cnn_shapes.py
```

这一章的达标标准：给你输入 `[B, 3, 64, 64]` 和类别数 7，你能不用复制现成类，写出一个输出 `[B, 7]` 的 CNN，并用 dummy batch 验证。

---

## 3. 数据管线：Dataset 给单样本，DataLoader 组 batch

### 3.1 分工

- `Dataset.__len__()`：有多少样本；
- `Dataset.__getitem__(index)`：返回**一个**样本；
- `DataLoader`：采样、打乱、并行读取、拼成 batch。

最简单的数据集：

```python
from torch.utils.data import TensorDataset, DataLoader

x = torch.randn(1000, 20)
y = torch.randint(0, 4, (1000,))
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

xb, yb = next(iter(loader))
print(xb.shape, yb.shape)  # [64, 20] [64]
```

自定义数据集时，先单独检查三个样本，再交给 DataLoader：

```python
for index in [0, 1, len(dataset) - 1]:
    x_one, y_one = dataset[index]
    print(index, x_one.shape, x_one.dtype, y_one)
```

然后检查一个 batch：

```python
xb, yb = next(iter(loader))
assert xb.ndim == 2
assert yb.ndim == 1
assert xb.shape[0] == yb.shape[0]
assert xb.dtype == torch.float32
assert yb.dtype == torch.long
```

### 3.2 训练集和验证集的边界

只用训练集拟合参数；验证集用于模型选择。标准化统计量、词表、类别映射等也只能从训练集拟合，否则会发生数据泄漏。

训练 loader 通常 `shuffle=True`；验证/测试 loader 通常 `shuffle=False`。验证集没有必要为了“整齐”而丢掉最后一个不满 batch 的样本。

### 3.3 `num_workers` 从 0 开始

先用 `num_workers=0` 保证报错栈清楚。功能正确后再增大并测吞吐量。多进程数据加载的最佳值取决于系统、数据解码开销和存储速度，不是越大越好。

---

## 4. 训练循环：每一行都有不可替代的职责

先完整看一遍，再拆开理解：

```python
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = y.shape[0]
        loss_sum += loss.detach().item() * batch_size
        correct += (logits.detach().argmax(dim=1) == y).sum().item()
        sample_count += batch_size

    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
    }
```

### 4.1 `model.train()` 不等于“开启梯度”

`model.train()` 让 Dropout、BatchNorm 等层进入训练行为；autograd 是否记录计算由 grad mode 控制。这是两套不同开关。

同理，`model.eval()` 也不会自动关闭梯度。因此验证时两者都要写：

```python
def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            batch_size = y.shape[0]
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == y).sum().item()
            sample_count += batch_size

    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
    }
```

### 4.2 为什么 loss 要乘 batch size

大部分 loss 默认返回当前 batch 的平均值。最后一个 batch 可能更小，直接对“各 batch 平均 loss”再平均会给小 batch 过高权重。正确的 epoch 平均是：

```python
total_loss += batch_mean_loss * batch_size
epoch_loss = total_loss / total_samples
```

### 4.3 梯度清零的位置

下面两种顺序都能工作：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

或本教程采用的：

```python
loss = loss_fn(model(x), y)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

关键是不允许上一个 batch 的梯度意外进入下一个 batch。只有明确做“梯度累积”时才故意延迟清零和更新。

### 4.4 梯度裁剪不是默认止痛药

确有梯度爆炸时可以在 `backward()` 后、`step()` 前：

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

但先找 NaN、学习率、输入尺度和不稳定运算的根因。盲目裁剪可能只是在掩盖问题。

完整实验：

```bash
python3 labs/03_train_classifier.py --device cpu --epochs 8
```

它使用合成的三分类数据，不需要联网。打开该文件，标出“数据、模型、loss、optimizer、train、evaluate、checkpoint”七块，再从空白文件写一遍。

---

## 5. 设备、保存与恢复

### 5.1 设备选择

```python
def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

模型和参与同一次运算的张量必须在同一设备：

```python
model = model.to(device)
x = x.to(device)
y = y.to(device)
```

通常不需要把整个 Dataset 预先搬到 GPU；逐 batch 搬运更节省显存。

### 5.2 保存 `state_dict`，而不是依赖整个 Python 对象

用于推理的最小保存：

```python
torch.save(model.state_dict(), "model.pt")
```

继续训练需要更多状态：

```python
torch.save(
    {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    },
    "checkpoint.pt",
)
```

恢复：

```python
checkpoint = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
start_epoch = checkpoint["epoch"] + 1
best_val_loss = checkpoint["best_val_loss"]
```

加载权重后用于推理，仍要调用 `model.eval()`。

注意：checkpoint 会反序列化数据，只加载可信来源的文件。新版本 PyTorch 对纯权重加载提供了更严格的 `weights_only` 行为；跨版本交付时应查当前 `torch.load` 文档，并在文件里记录 PyTorch 版本和模型配置。

---

## 6. 调试：固定顺序比“凭感觉改学习率”有效

遇到训练失败时，按下面顺序。不要同时改五个地方。

### 第 1 步：缩小到 CPU + 一个 batch

```python
x, y = next(iter(train_loader))
x, y = x.cpu(), y.cpu()
model = model.cpu()
```

CPU 报错通常更直接。先让一次 forward/backward 工作，再恢复完整数据和加速设备。

### 第 2 步：检查 shape、dtype、device、数值

```python
def inspect_tensor(name, value):
    print(
        name,
        "shape=", tuple(value.shape),
        "dtype=", value.dtype,
        "device=", value.device,
        "finite=", torch.isfinite(value).all().item()
              if value.is_floating_point() else "n/a",
    )
```

在 loss 前断言契约：

```python
assert logits.ndim == 2
assert y.ndim == 1
assert logits.shape[0] == y.shape[0]
assert y.dtype == torch.long
assert logits.device == y.device
assert torch.isfinite(logits).all()
assert 0 <= y.min() and y.max() < logits.shape[1]
```

### 第 3 步：强制过拟合一个 batch

这是最有价值的训练管线单元测试：

```python
x, y = next(iter(train_loader))
x, y = x.to(device), y.to(device)

for step in range(300):
    logits = model(x)
    loss = loss_fn(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item(), (model(x).argmax(1) == y).float().mean().item())
```

如果容量足够的模型连一个小 batch 都记不住，优先怀疑：

- 标签或 loss 搭配错误；
- 参数没有交给 optimizer；
- `detach()` / `.item()` 切断了图；
- 忘了 `backward()` 或 `step()`；
- 学习率极端；
- 数据与标签错位；
- 模型输出维度错误。

能过拟合一个 batch，只说明基本管线可学习，不代表泛化良好。

### 第 4 步：检查参数是否真的在更新

```python
before = model.net[0].weight.detach().clone()

loss.backward()
optimizer.step()

after = model.net[0].weight.detach()
print("parameter changed:", not torch.equal(before, after))
```

检查梯度：

```python
for name, parameter in model.named_parameters():
    if parameter.grad is None:
        print(name, "NO GRAD")
    else:
        print(name, parameter.grad.norm().item())
```

少数参数没有梯度可能符合设计；所有参数都没有梯度几乎一定是图断了或没有 backward。

### 第 5 步：定位 NaN/Inf

```python
torch.autograd.set_detect_anomaly(True)
```

它能把 backward 的异常追溯到对应 forward 操作，但会明显变慢，只在调试时开启。还要检查：

- 输入、logits、loss 从哪一步第一次变成非有限值；
- 学习率是否过大；
- 是否有除零、`log(0)`、大指数；
- 混合精度关闭后问题是否消失；
- 标签范围是否有效。

### 第 6 步：再看训练策略和性能

功能正确后才调整学习率、batch size、正则化和模型容量。最后再用 profiler、DataLoader worker、AMP 或 `torch.compile` 优化速度。

运行刻意报错实验：

```bash
python3 labs/04_debug_lab.py
```

先预测每个 case 会报什么或出现什么静默错误，再看输出。

---

## 7. 从 MLP 迁移到 CNN：循环不变，契约改变

从表格分类换成图片分类，训练循环几乎不用动。变化集中在：

```text
MLP 输入: [B, F]
CNN 输入: [B, C, H, W]
分类输出: [B, num_classes]
```

这就是框架熟练度的核心：把“任务相关部分”和“训练基础设施”分开。

对序列模型同样如此：

```text
token 输入: [B, T]
embedding:  [B, T, D]
序列分类:   [B, C]
token 分类: [B, T, C]
```

每换一种网络，先写 shape contract，再写 `forward`，最后用 dummy batch 证明契约。不要一边接真实数据一边猜维度。

---

## 8. AMP 与 `torch.compile`：基础正确后再加

自动混合精度的核心是 forward/loss 放在 autocast 区域，必要时用 GradScaler 保护低精度梯度。不同设备和 PyTorch 版本的推荐入口会变化，因此真正启用前查当前官方 AMP 文档。排查 NaN 时先关闭 AMP 做对照。

`torch.compile(model)` 可能提升模型执行速度，但会增加编译开销，并可能让报错路径更复杂。建议流程：

1. eager 模式完成正确性测试；
2. 固定输入规模做基准；
3. 再启用 compile；
4. 比较吞吐、显存和数值，而不是只看一次运行耗时。

学习阶段不要让性能技巧遮住核心训练循环。

---

## 9. 你的“空白文件算法”

新建一个训练脚本时，不背完整模板，只按以下顺序生成：

1. 写输入、输出、标签 shape contract；
2. import，设随机种子和 device；
3. 构造 Dataset / DataLoader，打印一个 batch；
4. 写 `nn.Module`，用 dummy batch 断言输出；
5. 创建 loss 和 optimizer；
6. 写 `train_one_epoch`；
7. 从训练函数复制并删掉三行，得到 `evaluate`：删 `zero_grad`、`backward`、`step`，加 `eval` 与 `inference_mode`；
8. 先过拟合一个 batch；
9. 跑完整训练，按样本数聚合指标；
10. 保存最佳 checkpoint，并重新加载做一次推理。

如果你能解释这十步，就不再依赖模板；具体 API 忘记了，只需局部查文档。

---

## 10. 14 天刻意练习计划

每天 45～90 分钟。所有“默写”都从真正的空白文件开始。

| 天 | 训练 | 验收 |
|---|---|---|
| 1 | 张量 shape、dtype、device、广播 | 找出 5 个 shape bug |
| 2 | autograd、梯度累加、`no_grad` | 手算并验证简单梯度 |
| 3 | 独立写 MLP | dummy batch 输出正确 |
| 4 | 独立写 CNN | 能打印每层 shape |
| 5 | Dataset / DataLoader | 自定义 Dataset 正确组 batch |
| 6 | 默写训练循环 | 15 分钟内无参考写完 |
| 7 | 默写验证循环和指标 | 无梯度、指标加权正确 |
| 8 | checkpoint 保存/恢复 | 恢复后预测完全一致 |
| 9 | 故意制造 dtype/device/shape 错误 | 看报错能定位到契约 |
| 10 | 单 batch 过拟合测试 | 小 batch 接近 100% |
| 11 | NaN、无梯度、模式错误排查 | 按固定清单定位 |
| 12 | 把 MLP 换成 CNN | 训练循环一行不改 |
| 13 | 完成毕业题 | 不看参考代码 |
| 14 | 第二次从空白重写 | 30 分钟内完整跑通 |

记一张错误日志，每次只写四项：症状、最小复现、根因、以后加什么断言。你的调试能力会比单纯多看十个模型提升得更快。

---

## 11. 基础阶段毕业验收

去做 [exercises/README.md](exercises/README.md) 的毕业题。满足以下条件才算完成：

- 不复制 `labs/03_train_classifier.py`；
- 先写 shape contract；
- 模型能过拟合 32 个样本；
- 完整训练有 train/validation 指标；
- checkpoint 恢复后，同一输入的 logits 与保存前一致；
- 能解释 `train/eval`、grad mode、`zero_grad/backward/step` 分别控制什么；
- 故意把标签改成 `float` 后，能在一分钟内定位错误；
- 故意让预测 `[B, 1]` 与目标 `[B]` 做 MSE，能发现广播风险。

达到这些标准后，再去学习残差网络、注意力、分布式训练，会顺很多：那些是模型或规模的变化，底层的形状、梯度、循环和调试纪律没有变。

---

## 12. 官方资料索引

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [Autograd anomaly detection](https://docs.pytorch.org/docs/stable/autograd.html#debugging-and-anomaly-detection)
- [Automatic Mixed Precision recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

---

## 13. 第二阶段：把单文件脚本变成科研项目

完成基础毕业题后，进入 [科研代码工程化](ENGINEERING.md)。配套的 [工程化 CNN 项目](projects/engineered_cnn/README.md) 会把一个真实训练任务拆成：

```text
config.py       参数和命令行
data.py         数据契约与 DataLoader
model.py        CNN 与输出契约
engine.py       训练、验证、单 batch 测试
checkpoint.py   best/last 保存与恢复
utils.py        seed、device、logging、CSV
train.py        只负责组装
predict.py      加载最佳模型做推理
```

这一阶段不是学习更多网络层，而是保证实验满足：

- 超参数不散落在代码里；
- 终端输出和磁盘日志一致；
- 每轮指标能被程序读取；
- 最佳模型与最近训练状态分开；
- 中断后能继续 optimizer 状态；
- 更换数据或模型时不必重写训练引擎。

运行：

```bash
python3 projects/engineered_cnn/train.py --device cpu --epochs 8
python3 projects/engineered_cnn/predict.py \
  --checkpoint projects/engineered_cnn/artifacts/default/best.pt \
  --device cpu
```

工程化阶段的验收不是“目录很好看”，而是删除 `last.pt` 中的 optimizer state 后，你能准确解释为什么它还能推理，却不能等价地恢复训练。

---

## 14. 第三阶段：用 MiniGPT 迁移核心能力

[MiniGPT 项目](projects/mini_gpt/README.md) 用纯 PyTorch 实现字符级 Decoder-only Transformer。它不会让你成为 LLM 专家，但会验证你能否把相同的 PyTorch 基础迁移到序列模型。

新的 shape contract：

```text
tokens:  [B,T]
embed:   [B,T,D]
q/k/v:   [B,H,T,Dh]
scores:  [B,H,T,T]
logits:  [B,T,V]
targets: [B,T]
```

你将完成：

1. 字符 tokenizer 与 next-token Dataset；
2. token/position embedding；
3. Q、K、V 多头 reshape；
4. causal mask；
5. pre-norm 残差 Decoder block；
6. 展平后的 token-level CrossEntropy；
7. step-based 训练、梯度裁剪、验证和 checkpoint；
8. temperature/top-k 自回归生成。

先用 20 steps smoke 配置验证管线，再训练 500 steps。生成文本是否流畅不是核心评分项；你能否解释每个维度、让一个 batch 过拟合并定位 mask 错误才是。

---

## 15. 从今天开始按什么顺序做

完整安排已经写入 [NEXT_STEPS.md](NEXT_STEPS.md)：

```text
第 1 周  默写基础循环与调试
第 2 周  独立完成 MLP 项目
第 3 周  CNN + 科研工程化
第 4 周  attention 与 Transformer 形状
第 5 周  训练、恢复和生成 MiniGPT
第 6 周  脱离教程完成真实数据项目
```

总验收命令：

```bash
python3 smoke_check.py
```

它只证明所有参考代码在当前环境能够运行。真正的学习验收仍然是：关掉参考实现，从空白文件重写。

---

## 16. 第四阶段：预训练模型与研究工具链

完成 MiniGPT 后，进入 [第三层研究工具链](RESEARCH_TOOLCHAIN.md)。MiniGPT 教你 Transformer 内部怎样工作；Hugging Face 工具链教你怎样可靠地使用、微调和管理现有预训练模型。

配套项目：

- [Transformers + Datasets + LoRA 文本分类](projects/hf_lora_text_classification/README.md)；
- [DDP + DeepSpeed](projects/distributed_training/README.md)；
- [Git/GitHub 实验管理](GIT_EXPERIMENTS.md)。

学习顺序：

```text
第 7 周  Transformers + Datasets + 自定义循环
第 8 周  Trainer + PEFT/LoRA 对照
第 9 周  W&B + Git/GitHub 实验谱系
第 10 周 DDP 进程语义 + DeepSpeed ZeRO 配置
```

第三方依赖使用独立环境：

```bash
python3.11 -m venv .venv-toolchain
source .venv-toolchain/bin/activate
python -m pip install -r requirements-toolchain.in
```

基础环境没有安装这些依赖，因此 `research_toolchain_check.py` 会验证源码语法、本地 CSV、DDP 和 DeepSpeed JSON；真正的 HF 模型下载与 LoRA 训练在新环境中执行。
