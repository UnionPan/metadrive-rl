"""Dense lead-gap dataset: one row per saved camera frame, any rollout source.

Reads rollouts collected with ``image_every_step`` (frames named
``<decision_id>_<step>.jpg``), labels each frame with the logged gap to the
lead vehicle at its time (privileged, training only), and assigns the split
from the frozen splits file by scene.

Usage:
    python -m metadrive_rl.data.build_gap_dataset <out-dir> <collection-dir>... \\
        --splits configs/evaluation/splits.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from .asl import find_rollouts, read_rollout
from .lead_gap import NO_LEAD_M, actor_half_lengths, gap_at, lead_gaps


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("collections", nargs="+", type=Path)
    parser.add_argument("--splits", type=Path, default=Path("configs/evaluation/splits.yaml"))
    parser.add_argument(
        "--scene-dir", type=Path, default=Path.home() / "ext/alpasim/data/nre-artifacts/all-usdzs"
    )
    args = parser.parse_args()

    splits = yaml.safe_load(args.splits.read_text())
    split_of = {s["scene_id"]: name for name, scenes in splits.items() for s in scenes}
    uuid = {s["scene_id"]: s["uuid"] for scenes in splits.values() for s in scenes}
    rows = []
    for collection in args.collections:
        images = collection / "images"
        for path in find_rollouts(collection):
            log = read_rollout(path)
            ts, gaps = lead_gaps(
                path, actor_half_lengths(args.scene_dir / f"{uuid[log.scene_id]}.usdz")
            )
            for d in log.decisions:
                image = images / f"{d['decision_id']}_{d['step']}.jpg"
                if not image.exists():
                    continue
                rows.append(
                    {
                        "split": split_of[log.scene_id],
                        "source": collection.name,
                        "scene_id": log.scene_id,
                        "rollout_uuid": log.rollout_uuid,
                        "time_us": d["time_us"],
                        "image": str(image),
                        "speed_mps": d["speed_mps"],
                        "lead_gap_m": gap_at(ts, gaps, d["time_us"]),
                    }
                )
    df = pl.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"collections": [str(c) for c in args.collections]}
    for split in sorted(df["split"].unique()):
        part = df.filter(pl.col("split") == split)
        part.write_parquet(args.out_dir / f"{split}.parquet")
        g = part["lead_gap_m"].to_numpy()
        manifest[split] = {
            "frames": len(part),
            "scenes": part["scene_id"].n_unique(),
            "by_source": part.group_by("source").len().to_dicts(),
            "share_no_lead": float((g >= NO_LEAD_M).mean()),
            "share_gap_below_10m": float((g < 10).mean()),
        }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
