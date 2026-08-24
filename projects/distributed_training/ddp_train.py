"""Minimal DDP training with correct sampling, metrics, and rank-zero saving.

Runs directly as one process or through:
    torchrun --standalone --nproc_per_node=2 ddp_train.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed:
        return False, 0, 0, 1, torch.device("cpu")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        # CPU/Gloo is for learning process semantics, not speed.
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return True, rank, local_rank, world_size, device


def make_dataset(samples: int, features: int, classes: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, features, generator=generator)
    teacher = torch.randn(features, classes, generator=generator)
    logits = x @ teacher + 0.1 * torch.randn(samples, classes, generator=generator)
    y = logits.argmax(dim=1)
    return TensorDataset(x, y)


class Classifier(nn.Module):
    def __init__(self, features: int, classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, 64),
            nn.ReLU(),
            nn.Linear(64, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def aggregate_metrics(
    loss_sum: float,
    correct: int,
    sample_count: int,
    device: torch.device,
    distributed: bool,
) -> tuple[float, float]:
    totals = torch.tensor(
        [loss_sum, float(correct), float(sample_count)],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return (totals[0] / totals[2]).item(), (totals[1] / totals[2]).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "ddp_model.pt",
    )
    args = parser.parse_args()

    distributed, rank, local_rank, world_size, device = setup_distributed()
    try:
        # Same initialization on every rank; DDP also broadcasts rank-zero weights.
        torch.manual_seed(args.seed)
        dataset = make_dataset(args.samples, args.features, args.classes, args.seed)
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
            )
            if distributed
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=0,
        )
        plain_model = Classifier(args.features, args.classes).to(device)
        if distributed:
            model: nn.Module = DDP(
                plain_model,
                device_ids=[local_rank] if device.type == "cuda" else None,
            )
        else:
            model = plain_model
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        loss_fn = nn.CrossEntropyLoss()

        if rank == 0:
            global_batch = args.batch_size * world_size
            print(
                f"distributed={distributed} backend_device={device} "
                f"world_size={world_size} per_device_batch={args.batch_size} "
                f"global_batch={global_batch}"
            )

        for epoch in range(1, args.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
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

            global_loss, global_accuracy = aggregate_metrics(
                loss_sum,
                correct,
                sample_count,
                device,
                distributed,
            )
            if rank == 0:
                print(
                    f"epoch={epoch:02d} global_loss={global_loss:.4f} "
                    f"global_accuracy={global_accuracy:.1%}"
                )

        if distributed:
            dist.barrier()
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            unwrapped = model.module if isinstance(model, DDP) else model
            torch.save(
                {
                    "model_state": unwrapped.state_dict(),
                    "features": args.features,
                    "classes": args.classes,
                    "world_size": world_size,
                },
                args.output,
            )
            print(f"rank=0 saved checkpoint={args.output}")
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

