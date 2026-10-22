"""Offline, matched-budget comparison of history encoders (AIS study, M5).

Same data (expert decision sequences per rollout), frozen features, cost
objective (Huber on per-action ADE), optimiser, epochs and evaluation for all
variants in ``metadrive_rl.ais.history``. The AIS variant adds
``lambda_p * MSE(predicted, actual next observation embedding)``, with the
embedding projected onto a fixed 64-d PCA basis fitted on training features.
Reports validation regret and, for ``ais``, held-out transition error.

Usage:
    python -m metadrive_rl.training.ais_offline --data data/datasets/expert_v2 \\
        --encoder dinov2-base --kind ais --out runs/ais/<name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from metadrive_rl.ais.history import HistoryPolicy


def sequences(data_dir: Path, split: str, encoder: str) -> dict:
    """Pad each rollout's decisions into (B, T, ...) tensors, time-ordered."""
    df = pl.read_parquet(data_dir / f"{split}.parquet").with_row_index("row")
    feats = np.load(data_dir / f"{split}.{encoder}.npy")
    states = np.array(df["state"].to_list(), dtype=np.float32)
    ade = np.array(df["ade_m"].to_list(), dtype=np.float32)
    groups = [
        g["row"].to_list()
        for _, g in df.sort("rollout_uuid", "time_us").group_by("rollout_uuid", maintain_order=True)
    ]
    T = max(len(g) for g in groups)
    B, D = len(groups), feats.shape[1] + states.shape[1]
    obs, cost, img = (
        np.zeros((B, T, D), np.float32),
        np.zeros((B, T, ade.shape[1]), np.float32),
        np.zeros((B, T, feats.shape[1]), np.float32),
    )
    mask = np.zeros((B, T), np.float32)
    for b, rows in enumerate(groups):
        obs[b, : len(rows)] = np.concatenate([feats[rows], states[rows]], 1)
        img[b, : len(rows)] = feats[rows]
        cost[b, : len(rows)] = ade[rows]
        mask[b, : len(rows)] = 1
    return {
        "obs": torch.tensor(obs),
        "cost": torch.tensor(cost),
        "img": img,
        "mask": torch.tensor(mask),
    }


def frame_ambiguity(train: dict, data: dict, k: int = 20) -> torch.Tensor:
    """Per position: how much the expert labels of the k most similar *training
    frames* (cosine similarity of the current image embedding alone) disagree,
    1 - share of the most common label. High values mark decisions the current
    frame does not determine, where history should matter."""

    def flat(d):
        m = d["mask"].bool()
        return F.normalize(torch.as_tensor(d["img"], device=m.device)[m], dim=-1), m

    tr, tr_mask = flat(train)
    tr_labels = train["cost"][tr_mask].argmin(-1)
    q, q_mask = flat(data)
    nn_idx = (q @ tr.T).topk(k, dim=-1).indices
    labels = tr_labels[nn_idx]  # (N, k)
    counts = F.one_hot(labels, train["cost"].shape[-1]).sum(1)
    amb = torch.zeros(q_mask.shape, device=q_mask.device)
    amb[q_mask] = 1.0 - counts.max(-1).values.float() / k
    return amb


def evaluate(model: HistoryPolicy, data: dict, proj: torch.Tensor | None) -> dict:
    model.eval()
    with torch.no_grad():
        pred, z = model(data["obs"], data["mask"])
        m = data["mask"].bool()
        choice = pred.argmin(-1)
        regret = data["cost"].gather(-1, choice[..., None])[..., 0] - data["cost"].min(-1).values
        out = {"regret_mean_m": float(regret[m].mean())}
        if "ambiguous" in data:
            amb = data["ambiguous"] & m
            out["regret_ambiguous_m"] = float(regret[amb].mean())
            out["regret_unambiguous_m"] = float(regret[m & ~amb].mean())
        if model.kind == "ais":
            out["transition_mse"] = float(_transition_loss(model, z, data, proj))
    return out


def _transition_loss(
    model: HistoryPolicy, z: torch.Tensor, data: dict, proj: torch.Tensor
) -> torch.Tensor:
    # Condition on the expert action (argmin cost) taken at t; predict e_{t+1}.
    action = data["cost"].argmin(-1)
    target = data["img_proj"][:, 1:]
    valid = (data["mask"][:, 1:] * data["mask"][:, :-1]).bool()
    pred = model.predict_next(z[:, :-1], action[:, :-1])
    return F.mse_loss(pred[valid], target[valid])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder", default="dinov2-base")
    parser.add_argument("--kind", choices=("frame", "stack", "gru", "ais"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lambda-p", type=float, default=1.0)
    parser.add_argument("--batch-rollouts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = sequences(args.data, "train", args.encoder)
    val = sequences(args.data, "val", args.encoder)
    # Fixed PCA basis of training image embeddings for the transition target.
    flat = train["img"][train["mask"].numpy().astype(bool)]
    mean = flat.mean(0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    proj = torch.tensor(vt[:64].T, device=device)
    for d in (train, val):
        d["img_proj"] = torch.tensor(d["img"] - mean, device=device) @ proj
        d["img_proj"] = d["img_proj"] / d["img_proj"][d["mask"].bool().to(device)].std()
        for k in ("obs", "cost", "mask"):
            d[k] = d[k].to(device)

    amb = frame_ambiguity(train, val)
    val["ambiguous"] = amb >= args.ambiguity_threshold
    model = HistoryPolicy(args.kind, train["obs"].shape[-1], args.hidden).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "metrics.jsonl", "w")
    best = (float("inf"), None)
    n = train["obs"].shape[0]
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_rollouts):
            idx = perm[i : i + args.batch_rollouts]
            batch = {k: v[idx] for k, v in train.items() if k != "img"}
            pred, z = model(batch["obs"], batch["mask"])
            m = batch["mask"].bool()
            loss = F.huber_loss(pred[m], batch["cost"][m])
            if args.kind == "ais":
                loss = loss + args.lambda_p * _transition_loss(model, z, batch, proj)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 9:
            row = {
                "epoch": epoch + 1,
                "train": evaluate(model, train, proj),
                "val": evaluate(model, val, proj),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            if row["val"]["regret_mean_m"] < best[0]:
                best = (row["val"]["regret_mean_m"], row)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "kind": args.kind,
                        "obs_dim": train["obs"].shape[-1],
                        "hidden": args.hidden,
                        "encoder": args.encoder,
                        "metrics": row,
                    },
                    args.out / "best.pt",
                )
    summary = {
        "kind": args.kind,
        "params": params,
        "best": best[1],
        "seed": args.seed,
        "val_ambiguous_fraction": float(val["ambiguous"][val["mask"].bool()].float().mean()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
