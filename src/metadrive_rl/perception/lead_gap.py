"""Lead-vehicle gap estimation from frozen camera features.

A small head regresses the bumper-to-bumper gap to the lead vehicle in the
ego corridor (0-60 m; 60 = no lead) from the same stacked image features the
policy uses plus ego speed. Trained on privileged simulator labels
(``metadrive_rl.data.lead_gap``); at drive time it only sees camera features.

Usage:
    python -m metadrive_rl.perception.lead_gap --data data/datasets/expert_v2 \\
        --encoder dinov2-base --context 3 --out runs/gap/<name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from metadrive_rl.data.lead_gap import NO_LEAD_M

SCALE_M = 10.0


class GapHead(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim + 1),
            nn.Linear(feature_dim + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.config = {"feature_dim": feature_dim, "hidden": hidden, "dropout": dropout}

    def forward(self, features: torch.Tensor, speed_mps: torch.Tensor) -> torch.Tensor:
        """Predicted gap in metres, clamped to [0, NO_LEAD_M]."""
        x = torch.cat([features, speed_mps[:, None] / 10.0], -1)
        return (self.net(x)[:, 0] * SCALE_M).clamp(0.0, NO_LEAD_M)


def load_gap_head(path: str | Path, device: str = "cpu") -> tuple[GapHead, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    head = GapHead(**ckpt["head_config"]).to(device).eval()
    head.load_state_dict(ckpt["state_dict"])
    return head, ckpt


def _split(data: Path, split: str, encoder: str, context: int, device: torch.device) -> dict:
    from metadrive_rl.training.sft import stack_context

    df = pl.read_parquet(data / f"{split}.parquet")
    feats = stack_context(df, np.load(data / f"{split}.{encoder}.npy"), context)
    speed = np.array(df["state"].to_list())[:, 0] * 10.0  # state[0] = speed / 10
    return {
        "x": torch.tensor(feats, device=device),
        "v": torch.tensor(speed, dtype=torch.float32, device=device),
        "y": torch.tensor(np.load(data / f"{split}.lead_gap.npy"), device=device),
    }


@torch.no_grad()
def metrics(head: GapHead, d: dict) -> dict:
    head.eval()
    p, y = head(d["x"], d["v"]), d["y"]
    close, near = y < 10.0, y < 30.0
    return {
        "mae_m": float((p - y).abs().mean()),
        "mae_lead_within_30m": float((p - y)[near].abs().mean()) if near.any() else None,
        "recall_lead_within_10m": float((p[close] < 10.0).float().mean()) if close.any() else None,
        "dangerous_miss_rate": float((p[close] > 20.0).float().mean()) if close.any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder", default="dinov2-base")
    parser.add_argument("--context", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = _split(args.data, "train", args.encoder, args.context, device)
    val = _split(args.data, "val", args.encoder, args.context, device)
    head = GapHead(train["x"].shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.out.mkdir(parents=True, exist_ok=True)
    best = (float("inf"), None)
    n = len(train["y"])
    for epoch in range(args.epochs):
        head.train()
        for idx in torch.randperm(n, device=device).split(128):
            loss = F.huber_loss(
                head(train["x"][idx], train["v"][idx]) / SCALE_M, train["y"][idx] / SCALE_M
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 9:
            row = {"epoch": epoch + 1, "train": metrics(head, train), "val": metrics(head, val)}
            if row["val"]["mae_m"] < best[0]:
                best = (row["val"]["mae_m"], row)
                torch.save(
                    {
                        "state_dict": head.state_dict(),
                        "head_config": head.config,
                        "encoder": args.encoder,
                        "context": args.context,
                        "metrics": row,
                    },
                    args.out / "best.pt",
                )
    (args.out / "summary.json").write_text(
        json.dumps({"best": best[1], "seed": args.seed}, indent=1)
    )
    print(json.dumps(best[1], indent=1))


if __name__ == "__main__":
    main()
