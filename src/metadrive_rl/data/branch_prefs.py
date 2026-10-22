"""Preference pairs from matched-state branch rollouts (scripts/collect_branches.sh).

Every action of a scene starts from the same branch state, so outcomes are
directly comparable. Explicit criteria, in order:

1. Safety: a rollout that passes (no at-fault collision, no offroad) beats one
   that fails.
2. Progress: between two passing rollouts, more relative progress wins if the
   difference exceeds ``margin``.

Rollouts that both fail give no pair. These are simulator-derived preferences,
not human feedback.

Usage:
    python -m metadrive_rl.data.branch_prefs <branch-dir> <out-dir> --sft <ckpt>

Writes ``outcomes.parquet`` (audit: one row per scene x action),
``states.parquet`` + ``features.npy`` (one branch state per scene, image
features stacked as the SFT checkpoint expects), and ``pairs.parquet``.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image

from metadrive_rl.perception import FrozenEncoder
from metadrive_rl.planner import HighLevelAction
from metadrive_rl.policy.features import state_features

from .asl import find_rollouts, read_rollout


def outcomes(branch_dir: Path) -> pl.DataFrame:
    rows = []
    for summary in sorted(branch_dir.glob("*/sim/aggregate/results-summary.json")):
        action = HighLevelAction.from_text(_label_to_action(summary.parts[-4])).to_text()
        for r in json.loads(summary.read_text())["rollouts"]:
            m = r["metrics"]
            rows.append(
                {
                    "scene_id": r["clipgt_id"],
                    "action": action,
                    "passed": bool(r["passed"]),
                    "score": r["score"],
                    "progress": m["progress"],
                    "collision_at_fault": m["collision_at_fault"],
                    "offroad": m["offroad"],
                }
            )
    return pl.DataFrame(rows)


def _label_to_action(label: str) -> str:
    """Directory labels replace '@' with '_' (CRUISE_10 -> CRUISE@10)."""
    head, _, tail = label.rpartition("_")
    return f"{head}@{tail}" if tail.isdigit() else label


def preference_pairs(out: pl.DataFrame, margin: float) -> pl.DataFrame:
    pairs = []
    for scene, group in out.group_by("scene_id"):
        rows = group.to_dicts()
        for a, b in itertools.permutations(rows, 2):
            safer = a["passed"] and not b["passed"]
            faster = a["passed"] and b["passed"] and a["progress"] > b["progress"] + margin
            if safer or faster:
                pairs.append(
                    {
                        "scene_id": scene[0] if isinstance(scene, tuple) else scene,
                        "winner": HighLevelAction.from_text(a["action"]).index,
                        "loser": HighLevelAction.from_text(b["action"]).index,
                        "reason": "safety" if safer else "progress",
                    }
                )
    return pl.DataFrame(pairs)


def branch_states(branch_dir: Path, branch_s: float, context: int) -> list[dict]:
    """Branch-state decision (and its context images) per scene, from one action run."""
    run = sorted(p for p in branch_dir.iterdir() if p.is_dir())[0]
    states = []
    for path in find_rollouts(run):
        log = read_rollout(path)
        decisions = [d for d in log.decisions if d["is_decision"]]
        t0 = decisions[0]["time_us"]
        k = next(i for i, d in enumerate(decisions) if d["time_us"] >= t0 + branch_s * 1e6)
        stack = [decisions[j] if j >= 0 else None for j in range(k, k - context, -1)]
        d = decisions[k]
        states.append(
            {
                "scene_id": log.scene_id,
                "images": [
                    str(run / "images" / f"{s['decision_id']}.jpg") if s else "" for s in stack
                ],
                "state": state_features(
                    d["speed_mps"], d["command"], d["lane_offset_m"], d.get("route_xy")
                ).tolist(),
            }
        )
    return states


@torch.no_grad()
def encode_states(states: list[dict], encoder: FrozenEncoder) -> np.ndarray:
    out = []
    for s in states:
        parts = []
        for path in s["images"]:
            if path:
                parts.append(encoder.encode([np.asarray(Image.open(path).convert("RGB"))])[0])
            else:
                parts.append(np.zeros(encoder.dim, dtype=np.float32))
        out.append(np.concatenate(parts))
    return np.stack(out).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("branch_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--sft", type=Path, required=True, help="defines encoder and context")
    parser.add_argument("--margin", type=float, default=0.02, help="min progress difference")
    args = parser.parse_args()

    ckpt = torch.load(args.sft, map_location="cpu", weights_only=True)
    cfg = json.loads((args.branch_dir / "branch.json").read_text())
    table = outcomes(args.branch_dir)
    pairs = preference_pairs(table, args.margin)
    states = branch_states(args.branch_dir, cfg["branch_s"], ckpt.get("context", 1))
    features = encode_states(states, FrozenEncoder(ckpt["encoder"], "cuda"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.write_parquet(args.out_dir / "outcomes.parquet")
    pl.DataFrame([{k: v for k, v in s.items() if k != "images"} for s in states]).write_parquet(
        args.out_dir / "states.parquet"
    )
    np.save(args.out_dir / "features.npy", features)
    pairs.write_parquet(args.out_dir / "pairs.parquet")
    summary = {
        "scenes": len(states),
        "outcomes": len(table),
        "pairs": len(pairs),
        "pairs_by_reason": pairs.group_by("reason").len().to_dicts() if len(pairs) else [],
        "pass_rate_by_action": table.group_by("action")
        .agg(pl.col("passed").mean())
        .sort("action")
        .to_dicts(),
        "encoder": ckpt["encoder"],
        "context": ckpt.get("context", 1),
        **cfg,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
