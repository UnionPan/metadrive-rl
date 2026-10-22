"""Lead-vehicle gap estimation from frozen camera features.

A small head regresses the bumper-to-bumper gap to the lead vehicle in the
ego corridor (0-60 m; 60 = no lead) from the same stacked image features the
policy uses plus ego speed. Trained on privileged simulator labels
(``metadrive_rl.data.lead_gap``); at drive time it only sees camera features.

Usage:
    python -m metadrive_rl.perception.lead_gap --data data/datasets/gap_dense \\
        --encoder dinov2-base-grid --context 1 --holdout-scenes 8 --out runs/gap/<name>
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


def _load(data: Path, split: str, encoder: str, context: int) -> tuple[pl.DataFrame, dict]:
    from metadrive_rl.training.sft import stack_context

    df = pl.read_parquet(data / f"{split}.parquet")
    feats = stack_context(df, np.load(data / f"{split}.{encoder}.npy"), context)
    if "lead_gap_m" in df.columns:  # dense gap dataset (build_gap_dataset)
        speed, gap = df["speed_mps"].to_numpy(), df["lead_gap_m"].to_numpy()
    else:  # decision dataset with a separate label file
        speed = np.array(df["state"].to_list())[:, 0] * 10.0  # state[0] = speed / 10
        gap = np.load(data / f"{split}.lead_gap.npy")
    return df, {"x": feats, "v": speed.astype(np.float32), "y": gap.astype(np.float32)}


def _splits(
    data: Path, encoder: str, context: int, holdout_scenes: int, device: torch.device
) -> tuple[dict, dict, list[str]]:
    """Train and val tensors; ``holdout_scenes`` train scenes (seeded) join val."""
    train_df, train = _load(data, "train", encoder, context)
    _, val = _load(data, "val", encoder, context)
    scenes = sorted(train_df["scene_id"].unique())
    held = sorted(np.random.default_rng(0).permutation(scenes)[:holdout_scenes].tolist())
    if held:
        mask = train_df["scene_id"].is_in(held).to_numpy()
        val = {k: np.concatenate([val[k], train[k][mask]]) for k in val}
        train = {k: v[~mask] for k, v in train.items()}
    to_t = lambda d: {k: torch.tensor(v, device=device) for k, v in d.items()}  # noqa: E731
    return to_t(train), to_t(val), held


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
    parser.add_argument(
        "--holdout-scenes", type=int, default=0, help="train scenes moved to validation"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, held = _splits(args.data, args.encoder, args.context, args.holdout_scenes, device)
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
        json.dumps(
            {"best": best[1], "seed": args.seed, "data": str(args.data), "holdout_scenes": held},
            indent=1,
        )
    )
    print(json.dumps(best[1], indent=1))


if __name__ == "__main__":
    main()
