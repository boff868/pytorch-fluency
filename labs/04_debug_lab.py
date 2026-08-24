"""实验 4：预测错误，再观察并解释。

每个 case 都在本进程内捕获，脚本本身应当成功退出。
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def run_case(
    name: str,
    function: Callable[[], None],
    *,
    expect_error: bool = True,
) -> None:
    print(f"\n[{name}]")
    try:
        function()
    except (AssertionError, RuntimeError, ValueError) as error:
        first_line = str(error).splitlines()[0]
        prefix = "caught" if expect_error else "UNEXPECTED"
        print(f"{prefix} {type(error).__name__}: {first_line}")
        if not expect_error:
            raise
    else:
        if expect_error:
            print("no exception: inspect the evidence for a silent bug")
        else:
            print("completed successfully")


def wrong_target_dtype() -> None:
    logits = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 2.0, 1.0])
    nn.CrossEntropyLoss()(logits, labels)


def broadcasting_bug() -> None:
    prediction = torch.randn(4, 1)
    target = torch.randn(4)
    difference = prediction - target
    print("prediction-target shape:", tuple(difference.shape))
    assert prediction.shape == target.shape, (
        f"MSE contract violated: {prediction.shape} != {target.shape}"
    )


def detached_loss() -> None:
    model = nn.Linear(2, 3)
    x = torch.randn(4, 2)
    y = torch.randint(0, 3, (4,))
    logits = model(x).detach()
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()


def missing_optimizer_step() -> None:
    torch.manual_seed(7)
    model = nn.Linear(2, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(8, 2)
    y = torch.randint(0, 3, (8,))
    before = model.weight.detach().clone()

    loss = nn.CrossEntropyLoss()(model(x), y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # optimizer.step() is intentionally missing.

    after = model.weight.detach()
    print("gradient exists:", model.weight.grad is not None)
    print("parameter changed:", not torch.equal(before, after))
    assert not torch.equal(before, after), "parameters did not change; was step() called?"


def correct_case() -> None:
    torch.manual_seed(7)
    model = nn.Linear(2, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(8, 2)
    y = torch.randint(0, 3, (8,))
    before = model.weight.detach().clone()

    logits = model(x)
    assert logits.shape == (8, 3)
    assert y.dtype == torch.long
    loss = nn.CrossEntropyLoss()(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert model.weight.grad is not None
    assert not torch.equal(before, model.weight.detach())
    print(f"loss={loss.item():.4f}, parameters updated=True")


def main() -> None:
    run_case("wrong target dtype", wrong_target_dtype)
    run_case("silent broadcasting", broadcasting_bug)
    run_case("detached computation graph", detached_loss)
    run_case("missing optimizer.step", missing_optimizer_step)
    run_case("correct train step", correct_case, expect_error=False)
    print("\nPASS: debug lab completed; explain every symptom before moving on")


if __name__ == "__main__":
    main()
