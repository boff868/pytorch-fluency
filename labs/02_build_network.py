"""实验 2：用 shape contract 搭 MLP，并确认参数已注册。"""

import torch
from torch import nn


class MLP(nn.Module):
    """Contract: [B, in_features] -> [B, num_classes] logits."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected [B, F], got {tuple(x.shape)}")
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    torch.manual_seed(7)
    model = MLP(in_features=12, hidden_features=32, num_classes=5)
    dummy = torch.randn(7, 12)
    logits = model(dummy)

    assert logits.shape == (7, 5)
    assert logits.requires_grad
    print(model)
    print(f"output shape: {tuple(logits.shape)}")
    print(f"trainable parameters: {count_parameters(model):,}")

    for name, parameter in model.named_parameters():
        print(f"{name:20} {tuple(parameter.shape)}")

    labels = torch.randint(0, 5, (7,), dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    print(f"loss: {loss.item():.4f}")
    print("PASS: model/shape lab")


if __name__ == "__main__":
    main()

