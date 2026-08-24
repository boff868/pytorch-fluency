"""实验 5：CNN 的 shape contract 与 forward hooks。"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Contract: [B, 3, H, W] -> [B, num_classes], H/W >= 4."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


def shape_hook(
    name: str,
) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
    def hook(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        print(
            f"{name:24} {tuple(inputs[0].shape)} -> {tuple(output.shape)}"
        )

    return hook


def main() -> None:
    torch.manual_seed(7)
    model = SmallCNN(num_classes=7)
    handles = []
    for name, module in model.named_modules():
        if name and not isinstance(module, nn.Sequential):
            handles.append(module.register_forward_hook(shape_hook(name)))

    dummy = torch.randn(5, 3, 64, 64)
    with torch.inference_mode():
        logits = model(dummy)

    for handle in handles:
        handle.remove()

    assert logits.shape == (5, 7)
    print("output contract satisfied:", tuple(logits.shape))

    # Adaptive pooling makes the classifier independent of H/W.
    second_dummy = torch.randn(2, 3, 40, 52)
    with torch.inference_mode():
        second_logits = model(second_dummy)
    assert second_logits.shape == (2, 7)
    print("different image size also works:", tuple(second_logits.shape))
    print("PASS: CNN shape lab")


if __name__ == "__main__":
    main()
