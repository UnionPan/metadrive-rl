"""M1 exit-gate analysis: do different fixed modes yield different executed paths?

Usage: python -m metadrive_rl.evaluation.mode_probe <probe-dir> <report-dir>

``<probe-dir>`` holds one sub-directory per policy spec, as written by
``scripts/mode_probe.sh``. Executed ego paths are compared with a reference
run at matched simulation timestamps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from metadrive_rl.data.asl import RolloutLog, find_rollouts, read_rollout  # noqa: E402
from metadrive_rl.evaluation.metrics import rollout_metrics  # noqa: E402

REFERENCE = "fixed_CRUISE_10"
METRICS = ("dist_traveled_m", "progress", "collision_any", "offroad", "wrong_lane")


def _complete_rollout(run_dir: Path) -> RolloutLog | None:
    """Longest rollout in the run (failed retries leave short logs behind)."""
    logs = [read_rollout(p) for p in find_rollouts(run_dir)]
    logs = [r for r in logs if len(r.ego_t_us) > 1]
    return max(logs, key=lambda r: len(r.ego_t_us), default=None)


def _divergence(a: RolloutLog, b: RolloutLog) -> tuple[float, float]:
    """(max, final) distance between two ego paths over shared timestamps."""
    common, ia, ib = np.intersect1d(a.ego_t_us, b.ego_t_us, return_indices=True)
    if len(common) == 0:
        return float("nan"), float("nan")
    d = np.linalg.norm(a.ego_xy[ia] - b.ego_xy[ib], axis=1)
    return float(d.max()), float(d[-1])


def _frenet(xy: np.ndarray, ref_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arc length along ``ref_xy`` and signed left offset for each point of ``xy``."""
    seg = np.diff(ref_xy, axis=0)
    seg_len = np.maximum(np.linalg.norm(seg, axis=1), 1e-9)
    s0 = np.concatenate([[0.0], np.cumsum(seg_len)])
    along, lateral = np.empty(len(xy)), np.empty(len(xy))
    for i, p in enumerate(xy):
        rel = p - ref_xy[:-1]
        u = np.clip((rel * seg).sum(1) / seg_len**2, 0, 1)
        d = np.linalg.norm(rel - u[:, None] * seg, axis=1)
        k = int(np.argmin(d))
        along[i] = s0[k] + u[k] * seg_len[k]
        lateral[i] = np.sign(seg[k, 0] * rel[k, 1] - seg[k, 1] * rel[k, 0]) * d[k]
    return along, lateral


def analyse(probe_dir: Path, report_dir: Path) -> str:
    runs = {}
    for run_dir in sorted(p for p in probe_dir.iterdir() if p.is_dir()):
        log = _complete_rollout(run_dir)
        if log is None:
            continue
        all_metrics = rollout_metrics(run_dir)
        metrics = all_metrics.filter(all_metrics["rollout_id"] == log.rollout_uuid)
        runs[run_dir.name] = (log, metrics.row(0, named=True) if len(metrics) else {})
    if REFERENCE not in runs:
        raise RuntimeError(f"reference run {REFERENCE} missing in {probe_dir}")
    ref = runs[REFERENCE][0]

    rows = []
    header = (
        "| run | actions | duration s | max div. vs CRUISE@10 m | final div. m | "
        + " | ".join(METRICS)
        + " |"
    )
    rows.append(header)
    rows.append("|" + "---|" * (header.count("|") - 1))
    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, (log, metrics) in runs.items():
        max_div, final_div = _divergence(log, ref)
        actions = sorted({d["action"] for d in log.decisions if d["is_decision"]})
        vals = []
        for m in METRICS:
            v = metrics.get(m)
            vals.append("—" if v is None else f"{v:.3g}")
        rows.append(
            f"| {name} | {', '.join(actions)} | {log.duration_s:.1f} | {max_div:.2f} | "
            f"{final_div:.2f} | " + " | ".join(vals) + " |"
        )
        t = (log.ego_t_us - log.ego_t_us[0]) / 1e6
        travelled = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(log.ego_xy, axis=0), axis=1))]
        )
        ax_s.plot(t, travelled, label=name, lw=1.5)
        along, lateral = _frenet(log.ego_xy, ref.ego_xy)
        ax_d.plot(along, lateral, label=name, lw=1.5)
    ax_s.set(xlabel="time (s)", ylabel="distance travelled (m)", title="Longitudinal: speed modes")
    ax_d.set(
        xlabel="distance along CRUISE@10 path (m)",
        ylabel="lateral offset, left + (m)",
        title="Lateral: lane-change modes",
    )
    ax_s.legend(fontsize=8)
    fig.suptitle(f"Executed ego paths per fixed action, {ref.scene_id} (closed loop)")
    report_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(report_dir / "m1_mode_probe_paths.png", dpi=120, bbox_inches="tight")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_dir", type=Path)
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()
    table = analyse(args.probe_dir, args.report_dir)
    (args.report_dir / "m1_mode_probe_table.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
