"""Build a labelled decision dataset from expert-collection runs.

Usage: python -m metadrive_rl.data.build_dataset <collection-dir> <out-dir> --split NAME

Writes ``<out-dir>/<split>.parquet`` (one row per labelled decision) and
``<out-dir>/<split>.manifest.json`` (sources, counts, label histogram).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from metadrive_rl.planner import ACTIONS
from metadrive_rl.policy.features import state_features

from .asl import find_rollouts, read_rollout
from .labels import expert_labels


def build(collection_dir: Path, split: str) -> tuple[pl.DataFrame, dict]:
    rows = []
    missing_images = 0
    for path in find_rollouts(collection_dir):
        log = read_rollout(path)
        routes = {d["decision_id"]: d.get("route_xy") for d in log.decisions if d["is_decision"]}
        for d in expert_labels(log):
            image = collection_dir / "images" / f"{d.decision_id}.jpg"
            if not image.exists():
                missing_images += 1
                continue
            rows.append(
                {
                    "decision_id": d.decision_id,
                    "split": split,
                    "scene_id": log.scene_id,
                    "rollout_uuid": log.rollout_uuid,
                    "time_us": d.time_us,
                    "image": str(image),
                    "state": state_features(
                        d.speed_mps, d.command, d.lane_offset_m, routes[d.decision_id]
                    ).tolist(),
                    "ade_m": d.ade_m.tolist(),
                    "label": d.label,
                }
            )
    df = pl.DataFrame(rows)
    labels = Counter(df["label"].to_list())
    manifest = {
        "split": split,
        "source": str(collection_dir),
        "rollouts": len(find_rollouts(collection_dir)),
        "scenes": sorted(df["scene_id"].unique().to_list()),
        "decisions": len(df),
        "missing_images": missing_images,
        "label_histogram": {ACTIONS[k].to_text(): labels.get(k, 0) for k in range(len(ACTIONS))},
        "median_best_ade_m": float(np.median([min(a) for a in df["ade_m"].to_list()])),
    }
    return df, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    df, manifest = build(args.collection_dir, args.split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out_dir / f"{args.split}.parquet")
    (args.out_dir / f"{args.split}.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
