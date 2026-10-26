"""Canonical episode records from AlpaSim rollouts (guideline §4).

One record per driver query, built only from log entries that precede the
query: the latest camera frame sent to the driver, the navigation command the
driver derived from the latest route (AlpaSim's own rule), and the executed ego
state. The trajectory the driver returned is attached as the issued action.
Per-timestep simulator metrics (``metrics.parquet``) are attached as outcome
targets at the query time; rewards must be computed from them, never fed back
as inputs.

Frames: positions are in AlpaSim's local frame (origin at the session start);
``issued_xy`` is in the rig frame at query time (x forward, y left).

Usage:
    python -m metadrive_ais.data.episodes <out-dir> <alpasim-run-dir>=<source> ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from metadrive_ais.data.asl import find_rollouts, iter_entries, poses_xyyaw

# Reward components copied from the per-timestep metrics (targets only).
OUTCOME_METRICS = (
    "collision_any",
    "collision_front",
    "collision_lateral",
    "collision_rear",
    "offroad",
    "wrong_lane",
    "progress",
    "dist_to_gt_trajectory",
    "min_distance_to_obstacle_m",
    "min_distance_to_lane_boundary_m",
)


@dataclass
class Step:
    time_us: int
    phase: str  # "forced_replay" or "policy"
    frame: bytes | None  # JPEG sent to the driver for this query
    frame_end_us: int | None
    command: str
    route_lateral_m: float | None  # route lateral offset at the lookahead point
    ego_xyyaw: tuple[float, float, float] | None
    issued_xy: list[list[float]] = field(default_factory=list)


def route_command(waypoints, threshold_m: float, lookahead_m: float) -> tuple[str, float | None]:
    """AlpaSim's ``determine_command_from_route`` on rig-frame waypoints."""
    if not waypoints:
        return "UNKNOWN", None
    for wp in waypoints:
        if np.hypot(wp.x, wp.y) >= lookahead_m:
            if wp.y > threshold_m:
                return "LEFT", float(wp.y)
            if wp.y < -threshold_m:
                return "RIGHT", float(wp.y)
            return "STRAIGHT", float(wp.y)
    return "STRAIGHT", None


def read_steps(
    path: str | Path, threshold_m: float = 3.0, lookahead_m: float = 20.0
) -> tuple[dict, list[Step]]:
    """Per-query records of one rollout log, plus its metadata."""
    meta: dict = {}
    executed: dict[int, np.ndarray] = {}
    frame = frame_end = None
    command, lateral = "UNKNOWN", None
    steps: list[Step] = []
    pending: Step | None = None
    for e in iter_entries(path):
        kind = e.WhichOneof("log_entry")
        if kind == "rollout_metadata":
            m = e.rollout_metadata
            meta = {
                "scene_id": m.session_metadata.scene_id,
                "rollout_id": m.session_metadata.session_uuid,
                "start_us": int(m.session_metadata.start_timestamp_us),
                "force_gt_us": int(m.force_gt_duration),
            }
        elif kind == "driver_camera_image":
            frame = e.driver_camera_image.camera_image.image_bytes
            frame_end = int(e.driver_camera_image.camera_image.frame_end_us)
        elif kind == "route_request":
            command, lateral = route_command(
                e.route_request.route.waypoints, threshold_m, lookahead_m
            )
        elif kind == "driver_ego_trajectory":
            for t, pose in zip(*poses_xyyaw(e.driver_ego_trajectory.trajectory), strict=True):
                executed[int(t)] = pose
        elif kind == "driver_request":
            now = int(e.driver_request.time_now_us)
            known = [t for t in executed if t <= now]
            ego = tuple(float(v) for v in executed[max(known)]) if known else None
            replay_end = meta.get("start_us", 0) + meta.get("force_gt_us", 0)
            pending = Step(
                time_us=now,
                phase="forced_replay" if now < replay_end else "policy",
                frame=frame if frame_end is not None and frame_end <= now else None,
                frame_end_us=frame_end if frame_end is not None and frame_end <= now else None,
                command=command,
                route_lateral_m=lateral,
                ego_xyyaw=ego,
            )
            steps.append(pending)
        elif kind == "driver_return" and pending is not None:
            t, poses = poses_xyyaw(e.driver_return.trajectory)
            if pending.ego_xyyaw is not None and len(t):
                x0, y0, yaw0 = pending.ego_xyyaw
                c, s = np.cos(yaw0), np.sin(yaw0)
                d = poses[:, :2] - (x0, y0)
                pending.issued_xy = (
                    np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)
                    .round(3)
                    .tolist()
                )
            pending = None
    return meta, steps


def outcomes_at(metrics: pl.DataFrame, times_us: list[int]) -> dict[str, list[float | None]]:
    """Latest valid value of each outcome metric at or before each time."""
    out = {}
    for name in OUTCOME_METRICS:
        m = metrics.filter((pl.col("name") == name) & pl.col("valid")).sort("timestamps_us")
        ts = m["timestamps_us"].cast(pl.Int64).to_numpy()
        vals = m["values"].to_numpy()
        idx = np.searchsorted(ts, np.asarray(times_us, dtype=np.int64), side="right") - 1
        out[name] = [float(vals[i]) if i >= 0 else None for i in idx]
    return out


def build(out_dir: Path, runs: list[tuple[Path, str]]) -> pl.DataFrame:
    rows = []
    for run, source in runs:
        for path in find_rollouts(run):
            meta, steps = read_steps(path)
            episode = f"{meta['scene_id']}/{meta['rollout_id']}"
            frames_dir = out_dir / "frames" / meta["rollout_id"]
            frames_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = path.parent / "metrics.parquet"
            outcomes = (
                outcomes_at(pl.read_parquet(metrics_file), [s.time_us for s in steps])
                if metrics_file.exists()
                else {name: [None] * len(steps) for name in OUTCOME_METRICS}
            )
            for i, s in enumerate(steps):
                image = None
                if s.frame is not None:
                    image = frames_dir / f"{i:03d}.jpg"
                    image.write_bytes(s.frame)
                rows.append(
                    {
                        "episode_id": episode,
                        "source": source,
                        "scene_id": meta["scene_id"],
                        "rollout_id": meta["rollout_id"],
                        "step": i,
                        "time_us": s.time_us,
                        "phase": s.phase,
                        "image": str(image) if image else None,
                        "frame_end_us": s.frame_end_us,
                        "command": s.command,
                        "route_lateral_m": s.route_lateral_m,
                        "ego_x": s.ego_xyyaw[0] if s.ego_xyyaw else None,
                        "ego_y": s.ego_xyyaw[1] if s.ego_xyyaw else None,
                        "ego_yaw": s.ego_xyyaw[2] if s.ego_xyyaw else None,
                        "issued_xy": s.issued_xy,
                        **{f"outcome_{k}": v[i] for k, v in outcomes.items()},
                    }
                )
    df = pl.DataFrame(rows)
    df.write_parquet(out_dir / "steps.parquet")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("runs", nargs="+", help="alpasim-run-dir=source-label")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = [(Path(r.partition("=")[0]), r.partition("=")[2] or Path(r).name) for r in args.runs]
    df = build(args.out_dir, runs)
    summary = df.group_by("source", "phase").agg(
        pl.col("episode_id").n_unique().alias("episodes"), pl.len().alias("steps")
    )
    print(summary.sort("source", "phase"))


if __name__ == "__main__":
    main()
