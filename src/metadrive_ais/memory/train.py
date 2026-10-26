"""Train an AIS memory (encoder + reward/next-embedding heads) on episode tables.

Episodes are split by scene: ``--holdout-scenes`` scenes (seeded) are held out
for validation. Reports held-out reward prediction (AUROC and Brier score for
collision/offroad, MAE for progress gain) and next-embedding prediction
(Gaussian NLL, cosine of the predicted mean).

Usage:
    python -m metadrive_ais.memory.train --data data/episodes/<name>/steps.parquet \\
        --encoder dinov3-small --kind gru --holdout-scenes 12 --seed 0 --out runs/ais/<name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from metadrive_ais.memory.ais import KINDS, AISHeads, MemoryEncoder, ais_losses
from metadrive_ais.memory.sequences import collate, episodes_from_table


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney AUROC; None without both classes."""
    pos, neg = labels > 0.5, labels <= 0.5
    if not pos.any() or not neg.any():
        return None
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores, kind="mergesort")] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


@torch.no_grad()
def evaluate(encoder, heads, batch) -> dict:
    encoder.eval(), heads.eval()
    out = heads(encoder(batch["embed"], batch["motion"]), batch["action"])
    losses = {k: float(v) for k, v in ais_losses(out, batch, heads.embed_norm).items()}
    m = batch["has_action"]
    report = {"loss": losses}
    for name in ("collision", "offroad"):
        target = batch[f"{name}_next"]
        ok = m & torch.isfinite(target)
        p = torch.sigmoid(out[f"{name}_logit"][ok]).cpu().numpy()
        y = target[ok].clamp(0, 1).cpu().numpy()
        report[name] = {
            "auroc": auroc(p, y),
            "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
            "positives": int((y > 0.5).sum()),
            "n": int(len(y)),
        }
    gain = batch["progress_gain"]
    ok = m & torch.isfinite(gain)
    report["progress_gain_mae"] = float((out["progress_gain"][ok] - gain[ok]).abs().mean())
    target = heads.embed_norm(batch["embed"][:, 1:])
    ok = m[:, :-1] & batch["valid"][:, 1:]
    cos = F.cosine_similarity(out["next_mean"][:, :-1][ok], target[ok], dim=-1)
    report["next_embed"] = {"nll": losses["next_embed"], "cosine": float(cos.mean())}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="steps.parquet tables")
    parser.add_argument("--encoder", default="dinov3-small")
    parser.add_argument("--kind", choices=KINDS, default="gru")
    parser.add_argument("--stack-k", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--holdout-scenes", type=int, default=12)
    parser.add_argument("--next-embed-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    episodes = []
    for table in args.data:
        steps = pl.read_parquet(table)
        feats = np.load(table.with_suffix("").as_posix() + f".{args.encoder}.npy")
        assert len(feats) == len(steps), "features out of sync with the episode table"
        episodes += episodes_from_table(steps, feats)
    scenes = sorted({e.scene_id for e in episodes})
    held = set(np.random.default_rng(0).permutation(scenes)[: args.holdout_scenes].tolist())
    train = [e for e in episodes if e.scene_id not in held]
    val = [e for e in episodes if e.scene_id in held]

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = episodes[0].embed.shape[1]
    encoder = MemoryEncoder(dim, args.kind, args.state_dim, args.stack_k).to(device)
    heads = AISHeads(args.state_dim, dim).to(device)
    params = [*encoder.parameters(), *heads.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    val_batch = collate(val, device)
    args.out.mkdir(parents=True, exist_ok=True)
    best, history = None, []
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        encoder.train(), heads.train()
        order = rng.permutation(len(train))
        for k in range(0, len(order), args.batch_episodes):
            batch = collate([train[i] for i in order[k : k + args.batch_episodes]], device)
            losses = ais_losses(
                heads(encoder(batch["embed"], batch["motion"]), batch["action"]),
                batch,
                heads.embed_norm,
            )
            loss = (
                losses["collision"]
                + losses["offroad"]
                + losses["progress"]
                + args.next_embed_weight * losses["next_embed"]
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if epoch % 10 == 9 or epoch == args.epochs - 1:
            report = evaluate(encoder, heads, val_batch)
            total = sum(report["loss"][k] for k in ("collision", "offroad", "progress")) + (
                args.next_embed_weight * report["loss"]["next_embed"]
            )
            history.append({"epoch": epoch + 1, "val_total": total, **report})
            if best is None or total < best[0]:
                best = (total, history[-1])
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "heads": heads.state_dict(),
                        "encoder_config": encoder.config,
                        "feature_encoder": args.encoder,
                        "epoch": epoch + 1,
                    },
                    args.out / "best.pt",
                )
    summary = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "episodes": {"train": len(train), "val": len(val)},
        "holdout_scenes": sorted(held),
        "best": best[1],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    (args.out / "history.jsonl").write_text("\n".join(json.dumps(h) for h in history) + "\n")
    print(json.dumps(summary["best"], indent=1))


if __name__ == "__main__":
    main()
