"""Open-loop driver error during AlpaSim's forced log replay.

While ``force_gt_duration_us`` lasts, the ego follows the human-driven path
but the driver is still queried. Each such prediction whose horizon ends
inside the replay can be scored against the human path: ADE/FDE and the signed
lateral error in the ego frame at query time (positive = left of the human).
A consistent lateral sign reveals a bias that does not depend on closed-loop
drift.

Usage: python -m metadrive_ais.evaluation.forced_replay <run-dir>[=label] ... [--force-gt-s 4]
"""

from __future__ import annotations

import argparse

import numpy as np

from metadrive_ais.data.asl import find_rollouts, iter_entries, poses_xyyaw


def replay_errors(path, force_gt_s: float) -> list[dict]:
    """One record per driver prediction fully inside the forced replay."""
    executed: dict[int, np.ndarray] = {}  # each message is a sliding history window
    predictions = []
    for e in iter_entries(path):
        kind = e.WhichOneof("log_entry")
        if kind == "driver_ego_trajectory":
            for ti, pi in zip(*poses_xyyaw(e.driver_ego_trajectory.trajectory), strict=True):
                executed[int(ti)] = pi
        elif kind == "driver_return" and len(e.driver_return.trajectory.poses) > 1:
            predictions.append(poses_xyyaw(e.driver_return.trajectory))
    if not executed:
        return []
    ego_t = np.array(sorted(executed), dtype=np.int64)
    ego = np.stack([executed[int(ti)] for ti in ego_t])
    ego[:, 2] = np.unwrap(ego[:, 2])
    end_us = ego_t[0] + force_gt_s * 1e6
    out = []
    for t, pred in predictions:
        if t[-1] > end_us or t[-1] > ego_t[-1]:
            continue
        human = np.stack([np.interp(t, ego_t, ego[:, i]) for i in range(2)], 1)
        x0, y0, yaw0 = (np.interp(t[0], ego_t, ego[:, i]) for i in range(3))
        c, s = np.cos(yaw0), np.sin(yaw0)
        d = pred[:, :2] - human
        lateral = -s * d[:, 0] + c * d[:, 1]
        longitudinal = c * d[:, 0] + s * d[:, 1]
        human_speed = np.linalg.norm(human[-1] - human[0]) / ((t[-1] - t[0]) / 1e6)
        out.append(
            {
                "ade_m": float(np.linalg.norm(d, axis=1).mean()),
                "fde_m": float(np.linalg.norm(d[-1])),
                "lateral_final_m": float(lateral[-1]),
                "longitudinal_final_m": float(longitudinal[-1]),
                "human_speed_mps": float(human_speed),
                "horizon_s": float((t[-1] - t[0]) / 1e6),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--force-gt-s", type=float, default=4.0)
    args = parser.parse_args()
    for spec in args.runs:
        path, _, label = spec.partition("=")
        rows = [r for p in find_rollouts(path) for r in replay_errors(p, args.force_gt_s)]
        if not rows:
            print(f"{label or path}: no predictions inside the forced replay")
            continue
        lat = np.array([r["lateral_final_m"] for r in rows])
        lon = np.array([r["longitudinal_final_m"] for r in rows])
        ade = np.mean([r["ade_m"] for r in rows])
        fde = np.mean([r["fde_m"] for r in rows])
        print(
            f"{label or path}: {len(rows)} predictions, horizon "
            f"{np.mean([r['horizon_s'] for r in rows]):.1f} s | ADE {ade:.2f} m, FDE {fde:.2f} m\n"
            f"  final lateral: mean {lat.mean():+.2f} m, median {np.median(lat):+.2f} m, "
            f"right of human {np.mean(lat < 0):.0%}\n"
            f"  final longitudinal: mean {lon.mean():+.2f} m, median {np.median(lon):+.2f} m, "
            f"ahead of human {np.mean(lon > 0):.0%} | human speed "
            f"{np.mean([r['human_speed_mps'] for r in rows]):.1f} m/s"
        )


if __name__ == "__main__":
    main()
