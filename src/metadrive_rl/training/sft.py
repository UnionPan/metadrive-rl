"""Supervised fine-tuning of the policy head on expert decisions.

Inputs are cached frozen-encoder features plus deployable state features.
Two objectives:

* ``soft_ce``: cross-entropy to soft labels ``softmax(-ADE / tau)``.
* ``ade_regression`` (default): regress every candidate's ADE (Huber) and
  act by argmin of the predicted cost. With little evidence the predictions
  shrink towards each action's mean cost, so the policy falls back to the
  best constant action instead of guessing expensive ones; it directly
  targets the selection metric.

The head's logits are ``-predicted ADE / tau`` for ``ade_regression``, so the
saved checkpoint is still a softmax policy. Checkpoints are selected by
validation regret: the ADE of the chosen action minus the best achievable
ADE (metres).

Usage:
    python -m metadrive_rl.training.sft --data data/datasets/expert_v1 \\
        --encoder dinov2-small --out runs/sft/<name>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from metadrive_rl.policy.neural import PolicyHead, save_checkpoint


def load_split(data_dir: Path, split: str, encoder: str, device: torch.device) -> dict:
    df = pl.read_parquet(data_dir / f"{split}.parquet")
    feats = np.load(data_dir / f"{split}.{encoder}.npy")
    assert len(feats) == len(df), "features out of sync with dataset rows"
    return {
        "features": torch.tensor(feats, device=device),
        "state": torch.tensor(np.array(df["state"].to_list()), dtype=torch.float32, device=device),
        "ade": torch.tensor(np.array(df["ade_m"].to_list()), dtype=torch.float32, device=device),
        "label": torch.tensor(df["label"].to_numpy(), device=device),
    }


def soft_targets(ade: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.softmax(-ade / tau, dim=-1)


def objective_loss(objective: str, logits: torch.Tensor, ade: torch.Tensor, tau: float) -> torch.Tensor:
    if objective == "soft_ce":
        return F.cross_entropy(logits, soft_targets(ade, tau))
    if objective == "ade_regression":
        return F.huber_loss(-logits * tau, ade)
    raise ValueError(objective)


@torch.no_grad()
def evaluate(head: PolicyHead, data: dict, tau: float, objective: str = "soft_ce") -> dict:
    head.eval()
    logits = head(data["features"], data["state"])
    pred = logits.argmax(-1)
    chosen_ade = data["ade"].gather(1, pred[:, None])[:, 0]
    regret = chosen_ade - data["ade"].min(-1).values
    return {
        "loss": float(objective_loss(objective, logits, data["ade"], tau)),
        "top1": float((pred == data["label"]).float().mean()),
        "regret_mean_m": float(regret.mean()),
        "regret_median_m": float(regret.median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder", default="dinov2-small")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--objective", choices=("ade_regression", "soft_ce"), default="ade_regression")
    parser.add_argument("--tau", type=float, default=0.5,
                        help="temperature (m): soft labels, or logits = -ADE/tau")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-image", action="store_true",
                        help="ablation: zero the image features (state-only policy)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_split(args.data, "train", args.encoder, device)
    val = load_split(args.data, "val", args.encoder, device)
    if args.no_image:
        for split in (train, val):
            split["features"].zero_()
    head = PolicyHead(train["features"].shape[1], args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    log = open(args.out / "metrics.jsonl", "w")
    best = (float("inf"), None)
    n = len(train["ade"])
    start = time.time()
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            logits = head(train["features"][idx], train["state"][idx])
            loss = objective_loss(args.objective, logits, train["ade"][idx], args.tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 9 or epoch == args.epochs - 1:
            row = {"epoch": epoch + 1, "train": evaluate(head, train, args.tau, args.objective),
                   "val": evaluate(head, val, args.tau, args.objective)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if row["val"]["regret_mean_m"] < best[0]:
                best = (row["val"]["regret_mean_m"], row)
                save_checkpoint(args.out / "best.pt", head, args.encoder,
                                f"sft:{args.out.name}:e{epoch + 1}", {"metrics": row})
    save_checkpoint(args.out / "last.pt", head, args.encoder, f"sft:{args.out.name}:last",
                    {"metrics": row})
    summary = {"best": best[1], "train_samples": n, "val_samples": len(val["label"]),
               "seconds": round(time.time() - start, 1),
               "peak_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
               if device.type == "cuda" else None}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
