"""Lead-vehicle gap labels from rollout logs (privileged; for training only).

At every logged frame, other actors are expressed in the ego box frame; the
lead vehicle is the nearest one ahead whose centre lies within a lane-wide
corridor. The gap is bumper to bumper: centre distance minus both
half-lengths (actor lengths from the scene's cuboid tracks). These labels
supervise a perception head; the policy never receives them as inputs.

Only actors the camera can show count. NuRec reconstructs vehicles only:
moving ones as dynamic Gaussians, parked ones inside the static background.
Pedestrians, riders, and moving vehicle tracks without Gaussians are never
rendered, so a camera-based gap cannot (and should not be asked to) see them.
"""

from __future__ import annotations

import io
import json
import zipfile
from functools import cache
from pathlib import Path

import numpy as np

from .asl import iter_entries

NO_LEAD_M = 60.0
CORRIDOR_HALF_WIDTH_M = 1.6
VEHICLE_CLASSES = frozenset(
    {"automobile", "heavy_truck", "bus", "trailer", "other_vehicle", "train_or_tram_car"}
)
# A track whose position spans less than this is parked: background, not dynamic.
STATIC_TRACK_M = 2.0


def visible_tracks(tracks_data: dict, rendered_spans: set[tuple[int, int]]) -> set[str]:
    """Track ids the renderer draws, from the tracks file and the time spans of
    the checkpoint's dynamic-Gaussian cuboids."""
    out = set()
    for tid, label, ts, poses in zip(
        tracks_data["tracks_id"],
        tracks_data["tracks_label_class"],
        tracks_data["tracks_timestamps_us"],
        tracks_data["tracks_poses"],
        strict=True,
    ):
        if label not in VEHICLE_CLASSES:
            continue
        xyz = np.asarray(poses, dtype=float)[:, :3]
        static = np.linalg.norm(xyz.max(0) - xyz.min(0)) < STATIC_TRACK_M
        if static or (int(ts[0]), int(ts[-1])) in rendered_spans:
            out.add(str(tid))
    return out


@cache
def actor_half_lengths(usdz: str | Path) -> dict[str, float]:
    """Half-lengths of the scene's camera-visible actors, by actor id."""
    from metadrive_rl.render.checkpoint import load_state_dict

    with zipfile.ZipFile(usdz) as z:
        tracks = next(iter(json.loads(z.read("sequence_tracks.json")).values()))
        sd = load_state_dict(io.BytesIO(z.read("checkpoint.ckpt")))
    prefix = "model.gaussians_nodes.dynamic_rigids."
    ranges = sd[prefix + "time_embed.timestamps_us_ranges"].tolist()
    counts = np.bincount(sd[prefix + "gaussian_cuboid_ids"].long().numpy(), minlength=len(ranges))
    spans = {(int(a), int(b)) for (a, b), n in zip(ranges, counts, strict=True) if n > 0}
    visible = visible_tracks(tracks["tracks_data"], spans)
    return {
        str(tid): dims[0] / 2
        for tid, dims in zip(
            tracks["tracks_data"]["tracks_id"],
            tracks["cuboidtracks_data"]["cuboids_dims"],
            strict=True,
        )
        if str(tid) in visible
    }


def _yaw(q) -> float:
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def lead_gaps(asl: str | Path, half_lengths: dict[str, float], ego_half_length: float = 2.6):
    """(timestamps_us, gap_m) for every logged frame; NO_LEAD_M when nothing is ahead.

    Only actors in ``half_lengths`` (the visible ones) can be a lead.
    """
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
            if a.actor_id not in half_lengths:
                continue
            dx, dy = a.actor_pose.vec.x - ex, a.actor_pose.vec.y - ey
            fwd, lat = c * dx + s * dy, -s * dx + c * dy
            if fwd > 0 and abs(lat) < CORRIDOR_HALF_WIDTH_M:
                g = fwd - ego_half_length - half_lengths[a.actor_id]
                gap = min(gap, max(g, 0.0))
        ts.append(e.actor_poses.timestamp_us)
        gaps.append(gap)
    return np.asarray(ts, dtype=np.int64), np.asarray(gaps)


def gap_at(ts: np.ndarray, gaps: np.ndarray, t_us: int) -> float:
    """Gap at the latest logged frame not after ``t_us``."""
    i = int(np.searchsorted(ts, t_us, side="right")) - 1
    return float(gaps[max(i, 0)])
