"""实验 1：shape、广播、autograd 与梯度累加。"""

import torch


def show(name: str, value: torch.Tensor) -> None:
    print(
        f"{name:>14} | shape={tuple(value.shape)!s:<12} "
        f"dtype={str(value.dtype):<14} device={value.device}"
    )


def shape_drill() -> None:
    print("\n[1] 形状操作")
    images = torch.randn(8, 3, 16, 16)
    show("images", images)
    show("flatten(1)", images.flatten(start_dim=1))
    show("mean H,W", images.mean(dim=(2, 3)))
    show("permute", images.permute(0, 2, 3, 1))


def broadcasting_trap() -> None:
    print("\n[2] 广播陷阱")
    prediction = torch.randn(8, 1)
    target = torch.randn(8)
    difference = prediction - target
    show("prediction", prediction)
    show("target", target)
    show("difference", difference)
    assert difference.shape == (8, 8)
    print("静默广播成了 [8, 8]；回归 loss 前应断言 prediction.shape == target.shape。")


def autograd_drill() -> None:
    print("\n[3] autograd 与梯度累加")
    weight = torch.tensor([2.0], requires_grad=True)
    x = torch.tensor([3.0])
    target = torch.tensor([1.0])

    loss = (weight * x - target).pow(2).mean()
    loss.backward()
    first_grad = weight.grad.detach().clone()
    print(f"loss={loss.item():.1f}, first grad={first_grad.item():.1f}")

    # 新前向产生新计算图；若不清空 .grad，第二次 backward 会继续累加。
    loss = (weight * x - target).pow(2).mean()
    loss.backward()
    second_grad = weight.grad.detach().clone()
    print(f"without zeroing, accumulated grad={second_grad.item():.1f}")
    assert torch.allclose(second_grad, 2 * first_grad)

    weight.grad = None
    loss = (weight * x - target).pow(2).mean()
    loss.backward()
    print(f"after clearing, grad={weight.grad.item():.1f}")
    assert torch.allclose(weight.grad, first_grad)


if __name__ == "__main__":
    torch.manual_seed(7)
    shape_drill()
    broadcasting_trap()
    autograd_drill()
    print("\nPASS: tensor/autograd lab")

