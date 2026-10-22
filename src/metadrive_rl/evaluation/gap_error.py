"""Closed-loop accuracy of the camera gap estimate that drove the governor.

Compares the ``lead_gap_est_m`` the driver logged at every call with the
privileged gap from the same rollout's actor poses.

Usage: python -m metadrive_rl.evaluation.gap_error <run-dir>... [--splits ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from metadrive_rl.data.asl import find_rollouts, read_rollout
from metadrive_rl.data.lead_gap import actor_half_lengths, gap_at, lead_gaps


def gap_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Error summary; the near-lead entries are what a governor depends on."""
    close, near, clear = true < 10.0, true < 30.0, true >= 30.0

    def frac(mask, cond):
        return float(cond[mask].mean()) if mask.any() else None

    return {
        "frames": int(len(true)),
        "mae_m": float(np.abs(pred - true).mean()),
        "mae_lead_within_30m": float(np.abs(pred - true)[near].mean()) if near.any() else None,
        "recall_lead_within_10m": frac(close, pred < 10.0),
        "dangerous_miss_rate": frac(close, pred > 20.0),
        "false_alarm_rate": frac(clear, pred < 10.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--splits", type=Path, default=Path("configs/evaluation/splits.yaml"))
    parser.add_argument(
        "--scene-dir", type=Path, default=Path.home() / "ext/alpasim/data/nre-artifacts/all-usdzs"
    )
    args = parser.parse_args()

    splits = yaml.safe_load(args.splits.read_text())
    uuid = {s["scene_id"]: s["uuid"] for scenes in splits.values() for s in scenes}
    for run in args.runs:
        pred, true = [], []
        for path in find_rollouts(run):
            log = read_rollout(path)
            ts, gaps = lead_gaps(
                path, actor_half_lengths(args.scene_dir / f"{uuid[log.scene_id]}.usdz")
            )
            for d in log.decisions:
                if d.get("lead_gap_est_m") is not None:
                    pred.append(d["lead_gap_est_m"])
                    true.append(gap_at(ts, gaps, d["time_us"]))
        if not pred:
            print(f"{run}: no gap estimates logged")
            continue
        print(run, json.dumps(gap_metrics(np.array(pred), np.array(true)), indent=1))


if __name__ == "__main__":
    main()
