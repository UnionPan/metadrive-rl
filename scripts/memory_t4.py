"""M4 memory-dependent test T4 (docs/PROGRESS.md): M2 AIS heads on informative steps.

Held-out CATK steps (the M2 split: 12 held-out train scenes) are split into
"informative" steps, where the ego's speed changed by more than 3 m/s over the
previous 2 s (4 queries) so history carries information the current frame and
last motion may not, and the rest. For each M2 checkpoint (frame / stack / GRU ×
3 seeds) the script reports next-step collision and offroad AUROC and the
next-embedding Gaussian NLL on each subset, with positive counts.

Usage: python scripts/memory_t4.py --out reports/m4_memory_t4.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from metadrive_ais.memory.ais import AISHeads, MemoryEncoder
from metadrive_ais.memory.sequences import collate, episodes_from_table
from metadrive_ais.memory.train import auroc

DATA = ["data/episodes/train_b0b_catk/steps.parquet", "data/episodes/train_b0a_catk/steps.parquet"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()

    episodes = []
    for table in map(Path, DATA):
        steps = pl.read_parquet(table)
        feats = np.load(table.with_suffix("").as_posix() + ".dinov3-small.npy")
        episodes += episodes_from_table(steps, feats)
    scenes = sorted({e.scene_id for e in episodes})
    held = set(np.random.default_rng(0).permutation(scenes)[:12].tolist())
    val = [e for e in episodes if e.scene_id in held]
    batch = collate(val)
    speed = batch["motion"][..., 4]
    change = torch.zeros_like(speed)
    change[:, 4:] = (speed[:, 4:] - speed[:, :-4]).abs()
    informative = change > args.threshold
    m = batch["has_action"]
    subsets = {"informative": m & informative, "rest": m & ~informative}

    result = {"threshold_mps": args.threshold, "models": {}}
    for kind in ("frame", "stack", "gru"):
        for seed in range(3):
            ckpt = torch.load(
                f"runs/ais/m2_{kind}_s{seed}/best.pt", map_location="cpu", weights_only=True
            )
            encoder = MemoryEncoder(**ckpt["encoder_config"]).eval()
            encoder.load_state_dict(ckpt["encoder"])
            heads = AISHeads(encoder.state_dim, batch["embed"].shape[-1]).eval()
            heads.load_state_dict(ckpt["heads"])
            with torch.no_grad():
                out = heads(encoder(batch["embed"], batch["motion"]), batch["action"])
                target = heads.embed_norm(batch["embed"][:, 1:])
                mean, log_var = out["next_mean"][:, :-1], out["next_log_var"][:, :-1]
                nll = 0.5 * (log_var + (target - mean) ** 2 / log_var.exp()).mean(-1)
            row = {}
            for name, sub in subsets.items():
                cell = {}
                for event in ("collision", "offroad"):
                    y = batch[f"{event}_next"]
                    ok = sub & torch.isfinite(y)
                    p = torch.sigmoid(out[f"{event}_logit"][ok]).numpy()
                    labels = y[ok].clamp(0, 1).numpy()
                    cell[event] = {
                        "auroc": auroc(p, labels),
                        "positives": int((labels > 0.5).sum()),
                        "n": int(len(labels)),
                    }
                ok = sub[:, :-1] & batch["valid"][:, 1:]
                cell["next_embed_nll"] = float(nll[ok].mean())
                cell["steps"] = int(sub.sum())
                row[name] = cell
            result["models"][f"{kind}_s{seed}"] = row
            print(
                kind,
                seed,
                {
                    k: (
                        v["collision"]["auroc"],
                        v["offroad"]["auroc"],
                        round(v["next_embed_nll"], 3),
                    )
                    for k, v in row.items()
                },
                flush=True,
            )
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
