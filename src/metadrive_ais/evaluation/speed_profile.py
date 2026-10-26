"""Closed-loop speed profile: executed vs planned ego speed after control starts.

A policy given its own executed speed can learn to keep it (the inertia or
copycat problem): fine on human replay, self-reinforcing in closed loop. This
reports, per run and per time bin since the forced replay ended, the mean
executed speed (between consecutive queries) and the mean speed of the issued
plan (path length over the plan horizon at 2 Hz), and how the planned speed
follows the current speed (least-squares slope over all policy queries).

Usage: python -m metadrive_ais.evaluation.speed_profile <run-dir>[=label] ...
"""

from __future__ import annotations

import argparse

import numpy as np

from metadrive_ais.data.asl import find_rollouts
from metadrive_ais.data.episodes import read_steps

BINS_S = (0, 4, 8, 12, 16)  # val_cl scenes give ~16 s of control
PLAN_DT_S = 0.5


def rollout_speeds(path) -> list[tuple[float, float, float]]:
    """(seconds since control start, executed speed, planned speed) per policy query."""
    meta, steps = read_steps(path)
    start_us = meta.get("start_us", 0) + meta.get("force_gt_us", 0)
    out = []
    for prev, cur in zip(steps, steps[1:], strict=False):
        if cur.phase != "policy" or prev.ego_xyyaw is None or cur.ego_xyyaw is None:
            continue
        if not cur.issued_xy:
            continue
        dt = max((cur.time_us - prev.time_us) / 1e6, 1e-3)
        executed = (
            np.hypot(cur.ego_xyyaw[0] - prev.ego_xyyaw[0], cur.ego_xyyaw[1] - prev.ego_xyyaw[1])
            / dt
        )
        plan = np.vstack([[0.0, 0.0], np.asarray(cur.issued_xy)])
        planned = np.linalg.norm(np.diff(plan, axis=0), axis=1).sum() / (
            PLAN_DT_S * (len(plan) - 1)
        )
        out.append(((cur.time_us - start_us) / 1e6, float(executed), float(planned)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    header = " | ".join(f"{a}–{b} s" for a, b in zip(BINS_S, BINS_S[1:], strict=False))
    print(f"| run | {header} | planned-on-current slope |")
    print("|---|" + "---|" * (len(BINS_S) - 1) + "---|")
    for spec in args.runs:
        path, _, label = spec.partition("=")
        rows = np.array([r for p in find_rollouts(path) for r in rollout_speeds(p)])
        cells = []
        for a, b in zip(BINS_S, BINS_S[1:], strict=False):
            m = (rows[:, 0] >= a) & (rows[:, 0] < b)
            cells.append(f"{rows[m, 1].mean():.1f} / {rows[m, 2].mean():.1f}" if m.any() else "–")
        slope = np.polyfit(rows[:, 1], rows[:, 2], 1)[0]
        print(f"| {label or path} | " + " | ".join(cells) + f" | {slope:.2f} |")
    print("\nCells: mean executed / planned speed (m/s) over policy queries in the bin.")


if __name__ == "__main__":
    main()
