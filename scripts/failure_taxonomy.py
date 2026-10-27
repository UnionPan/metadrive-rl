"""Failure taxonomy of closed-loop runs from AlpaSim's scored results (guideline M6).

Per run: rollouts by outcome (pass, offroad, at-fault collision by type,
simulator abort), time from control start to the end of the scored window
(the failure time for failed rollouts), wrong-lane share, and distance from the
logged human trajectory. All values are AlpaSim's own per-rollout aggregates
(``aggregate/results-summary.json``).

Usage: python scripts/failure_taxonomy.py <run-dir>=<label> ...
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def classify(r: dict) -> str:
    m = r["metrics"]
    if "error" in m:
        return "simulator abort"
    reason = r.get("failure_reason")
    if reason is None:
        return "pass"
    if reason == "offroad":
        return "offroad"
    kinds = [k for k in ("front", "lateral", "rear") if m.get(f"collision_{k}", 0) > 0]
    return "collision (" + "+".join(kinds or ["?"]) + ")"


ORDER = [
    "pass",
    "offroad",
    "collision (front)",
    "collision (lateral)",
    "collision (front+lateral)",
    "simulator abort",
]
for spec in sys.argv[1:]:
    run, _, label = spec.partition("=")
    rollouts = json.loads(Path(run, "sim/aggregate/results-summary.json").read_text())["rollouts"]
    classes = [classify(r) for r in rollouts]
    counts = Counter(classes)
    n = len(rollouts)
    print(f"## {label or run}: {n} rollouts, {len({r['clipgt_id'] for r in rollouts})} scenes\n")
    print("| outcome | rollouts | share | median time to end of scored window (s) |")
    print("|---|---|---|---|")
    for cls in ORDER + sorted(set(counts) - set(ORDER)):
        if not counts.get(cls):
            continue
        t = [
            20 * r["metrics"]["duration_frac_20s"]
            for r, c in zip(rollouts, classes, strict=True)
            if c == cls and "error" not in r["metrics"]
        ]
        med = f"{np.median(t):.1f}" if t else "–"
        print(f"| {cls} | {counts[cls]} | {counts[cls] / n:.0%} | {med} |")
    scored = [r for r in rollouts if "error" not in r["metrics"]]
    failed = [r for r in scored if r.get("failure_reason")]
    wrong_lane = np.mean([r["metrics"].get("wrong_lane", 0) > 0 for r in scored])
    gt_dist = [r["metrics"]["dist_to_gt_trajectory"] for r in failed]
    print(
        f"\nWrong lane at some point: {wrong_lane:.0%} of scored rollouts. "
        f"Distance from the human trajectory when failing: median {np.median(gt_dist):.1f} m.\n"
    )
