"""Lead-vehicle gap labels from rollout logs (privileged; for training only).

At every logged frame, other actors are expressed in the ego box frame; the
lead vehicle is the nearest one ahead whose centre lies within a lane-wide
corridor. The gap is bumper to bumper: centre distance minus both
half-lengths (actor lengths from the scene's cuboid tracks). These labels
supervise a perception head; the policy never receives them as inputs.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from .asl import iter_entries

NO_LEAD_M = 60.0
CORRIDOR_HALF_WIDTH_M = 1.6


def actor_half_lengths(usdz: str | Path) -> dict[str, float]:
    with zipfile.ZipFile(usdz) as z:
        tracks = next(iter(json.loads(z.read("sequence_tracks.json")).values()))
    return {
        str(tid): dims[0] / 2
        for tid, dims in zip(
            tracks["tracks_data"]["tracks_id"],
            tracks["cuboidtracks_data"]["cuboids_dims"],
            strict=True,
        )
    }


def _yaw(q) -> float:
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def lead_gaps(asl: str | Path, half_lengths: dict[str, float], ego_half_length: float = 2.6):
    """(timestamps_us, gap_m) for every logged frame; NO_LEAD_M when nothing is ahead."""
    ts, gaps = [], []
    for e in iter_entries(asl):
        if e.WhichOneof("log_entry") != "actor_poses":
            continue
        actors = e.actor_poses.actor_poses
        ego = next((a for a in actors if a.actor_id == "EGO"), None)
        if ego is None:
            continue
        yaw = _yaw(ego.actor_pose.quat)
        c, s = np.cos(yaw), np.sin(yaw)
        ex, ey = ego.actor_pose.vec.x, ego.actor_pose.vec.y
        gap = NO_LEAD_M
        for a in actors:
            if a.actor_id == "EGO":
                continue
            dx, dy = a.actor_pose.vec.x - ex, a.actor_pose.vec.y - ey
            fwd, lat = c * dx + s * dy, -s * dx + c * dy
            if fwd > 0 and abs(lat) < CORRIDOR_HALF_WIDTH_M:
                g = fwd - ego_half_length - half_lengths.get(a.actor_id, 2.3)
                gap = min(gap, max(g, 0.0))
        ts.append(e.actor_poses.timestamp_us)
        gaps.append(gap)
    return np.asarray(ts, dtype=np.int64), np.asarray(gaps)


def gap_at(ts: np.ndarray, gaps: np.ndarray, t_us: int) -> float:
    """Gap at the latest logged frame not after ``t_us``."""
    i = int(np.searchsorted(ts, t_us, side="right")) - 1
    return float(gaps[max(i, 0)])
