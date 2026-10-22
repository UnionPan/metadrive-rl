"""Summarise closed-loop evaluations from AlpaSim's aggregated results.

AlpaSim's aggregation scores each rollout up to its first incident: an
at-fault collision or offroad fails the rollout, otherwise the score is the
clipped relative progress (see ``score_criteria`` in results-summary.json).
This reports per-run means with 95% confidence intervals from a cluster
bootstrap over scenes: rollouts of one scene are not independent, so scenes
are resampled and each contributes the mean of its rollouts. Rollouts aborted
by the simulator (e.g. CATK RPC failures) carry no metrics; they are excluded
and counted.

Usage:
    python -m metadrive_ais.evaluation.summarize <run-dir>[=label] ... \\
        [--common-scenes] [--paired LABEL_A LABEL_B] [--out table.md]
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


def simulator_error(rollout: dict) -> bool:
    """Whether a results-summary rollout was aborted by the simulator."""
    return "error" in rollout["metrics"]


def rollouts(run_dir: Path) -> tuple[list[dict], int]:
    """Scored rollouts of a run, and the number aborted by the simulator."""
    files = sorted(run_dir.glob("**/aggregate/results-summary.json"))
    if not files:
        raise FileNotFoundError(f"no results-summary.json under {run_dir}")
    out, aborted = [], 0
    for f in files:
        for r in json.loads(f.read_text())["rollouts"]:
            if simulator_error(r):
                aborted += 1
                continue
            out.append(
                {
                    "scene": r["clipgt_id"],
                    "score": r["score"],
                    "passed": float(r["passed"]),
                    **{k: r["metrics"][k] for k, _ in METRICS[2:]},
                }
            )
    return out, aborted


def bootstrap_ci(scene_values: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean over scenes, resampling scenes with replacement."""
    if len(scene_values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(scene_values, size=(n, len(scene_values)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def scene_means(rows: list[dict], key: str, scenes: list[str]) -> np.ndarray:
    return np.array([np.mean([r[key] for r in rows if r["scene"] == s]) for s in scenes])


def table(runs: dict[str, list[dict]], aborted: dict[str, int]) -> str:
    header = (
        "| run | scenes | rollouts | aborted | " + " | ".join(name for _, name in METRICS) + " |"
    )
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for label, rows in runs.items():
        scenes = sorted({r["scene"] for r in rows})
        cells = []
        for key, _ in METRICS:
            values = scene_means(rows, key, scenes)
            lo, hi = bootstrap_ci(values)
            mean = values.mean()
            cells.append(f"{mean:.3g} [{lo:.3g}, {hi:.3g}]" if np.isfinite(lo) else f"{mean:.3g}")
        lines.append(
            f"| {label} | {len(scenes)} | {len(rows)} | {aborted[label]} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def paired(a: list[dict], b: list[dict], label: str) -> str:
    """Per-scene paired difference a - b on the scenes both completed."""
    scenes = sorted({r["scene"] for r in a} & {r["scene"] for r in b})
    lines = [f"Paired {label} over {len(scenes)} matched scenes:"]
    for key, name in METRICS:
        diff = scene_means(a, key, scenes) - scene_means(b, key, scenes)
        lo, hi = bootstrap_ci(diff)
        changed = int((np.abs(diff) > 1e-9).sum())
        lines.append(
            f"- {name}: {diff.mean():+.3g} [{lo:+.3g}, {hi:+.3g}] ({changed} scenes differ)"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+", help="run directory, optionally run_dir=label")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--common-scenes",
        action="store_true",
        help="restrict every run to scenes that completed in all runs (matched comparison)",
    )
    parser.add_argument("--paired", nargs=2, metavar=("A", "B"), help="report A - B per scene")
    args = parser.parse_args()
    runs, aborted = {}, {}
    for spec in args.runs:
        path, _, label = spec.partition("=")
        label = label or Path(path).name
        runs[label], aborted[label] = rollouts(Path(path))
    if args.common_scenes:
        common = set.intersection(*({r["scene"] for r in rows} for rows in runs.values()))
        runs = {k: [r for r in rows if r["scene"] in common] for k, rows in runs.items()}
    text = table(runs, aborted)
    if args.paired:
        a, b = args.paired
        text += "\n\n" + paired(runs[a], runs[b], f"{a} − {b}")
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
