"""Read AlpaSim ASL rollout logs (size-prefixed ``LogEntry`` protobufs)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from alpasim_grpc.v0.logging_pb2 import LogEntry


def iter_entries(path: str | Path):
    with open(path, "rb") as f:
        while prefix := f.read(4):
            (size,) = struct.unpack(">L", prefix)
            chunk = f.read(size)
            if len(chunk) != size:  # truncated tail of an interrupted rollout
                return
            yield LogEntry.FromString(chunk)


def find_rollouts(run_dir: str | Path) -> list[Path]:
    """Completed rollout logs under an AlpaSim log directory."""
    return sorted(
        p
        for p in Path(run_dir).glob("**/rollouts/*/*/rollout.asl")
        if (p.parent / "_complete").exists()
    )


def poses_xyyaw(trajectory) -> tuple[np.ndarray, np.ndarray]:
    """(timestamps_us, [x, y, yaw]) of a grpc ``Trajectory``."""
    t = np.array([p.timestamp_us for p in trajectory.poses], dtype=np.int64)
    out = np.empty((len(t), 3))
    for i, p in enumerate(trajectory.poses):
        q = p.pose.quat
        out[i] = (
            p.pose.vec.x,
            p.pose.vec.y,
            np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
        )
    return t, out
