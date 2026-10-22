"""Summarise closed-loop evaluations from AlpaSim's aggregated results.

AlpaSim's aggregation scores each rollout up to its first incident: an
at-fault collision or offroad fails the rollout, otherwise the score is the
clipped relative progress (see ``score_criteria`` in results-summary.json).
This reports per-run means with bootstrap 95% confidence intervals.

Usage: python -m metadrive_rl.evaluation.summarize <run-dir>[=label] ... [--out table.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = (
    ("score", "score"),
    ("passed", "pass rate"),
    ("collision_at_fault", "at-fault collision"),
    ("offroad", "offroad"),
    ("progress", "progress"),
    ("dist_traveled_m", "distance (m)"),
)


def rollouts(run_dir: Path) -> list[dict]:
    files = sorted(run_dir.glob("**/aggregate/results-summary.json"))
    if not files:
        raise FileNotFoundError(f"no results-summary.json under {run_dir}")
    out = []
    for f in files:
        for r in json.loads(f.read_text())["rollouts"]:
            out.append({"scene": r["clipgt_id"], "score": r["score"], "passed": float(r["passed"]),
                        **{k: r["metrics"][k] for k, _ in METRICS[2:]}})
    return out


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def table(runs: dict[str, list[dict]]) -> str:
    header = "| run | rollouts | " + " | ".join(name for _, name in METRICS) + " |"
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for label, rows in runs.items():
        cells = []
        for key, _ in METRICS:
            v = np.array([r[key] for r in rows], dtype=float)
            lo, hi = bootstrap_ci(v)
            cells.append(f"{v.mean():.3g} [{lo:.3g}, {hi:.3g}]" if np.isfinite(lo) else f"{v.mean():.3g}")
        lines.append(f"| {label} | {len(rows)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run directory, optionally run_dir=label")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    runs = {}
    for spec in args.runs:
        path, _, label = spec.partition("=")
        runs[label or Path(path).name] = rollouts(Path(path))
    text = table(runs)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
