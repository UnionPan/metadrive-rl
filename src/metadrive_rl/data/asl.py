"""Read AlpaSim ASL rollout logs into compact numpy records.

An ASL file is a stream of size-prefixed ``LogEntry`` protobufs. We extract the
executed ego trajectory and the MetaDrive decision record that the driver
plugin embeds in every drive response (see ``metadrive_rl.alpasim.model``).
"""

from __future__ import annotations

import json
import pickle
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from alpasim_grpc.v0.logging_pb2 import LogEntry

from metadrive_rl.alpasim.model import RECORD_KEY


@dataclass
class RolloutLog:
    path: Path
    scene_id: str
    rollout_uuid: str
    ego_t_us: np.ndarray  # (N,)
    ego_xy: np.ndarray  # (N, 2) ego bounding-box centre, scene frame
    ego_yaw: np.ndarray  # (N,)
    # Ego rig poses (the frame planner trajectories are expressed in), from the
    # egomotion the runtime sends the driver.
    rig_t_us: np.ndarray  # (M,)
    rig_xy: np.ndarray  # (M, 2) scene frame
    # One MetaDrive record per drive response, in order, plus ``time_us`` (the
    # drive request time). Decision records also carry the planner candidates
    # as returned to the runtime: ``candidates_xy`` (K, T, 2) in the scene frame
    # and ``candidates_t_us`` (T,).
    decisions: list[dict]

    def rig_xy_at(self, t_us: np.ndarray) -> np.ndarray:
        """Executed rig position at ``t_us`` (linear interpolation), (N, 2)."""
        return np.stack([np.interp(t_us, self.rig_t_us, self.rig_xy[:, i]) for i in range(2)], -1)

    @property
    def duration_s(self) -> float:
        return float(self.ego_t_us[-1] - self.ego_t_us[0]) / 1e6 if len(self.ego_t_us) else 0.0

    @property
    def path_length_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.ego_xy, axis=0), axis=1).sum())

    def speeds(self) -> np.ndarray:
        dt = np.diff(self.ego_t_us) / 1e6
        return np.linalg.norm(np.diff(self.ego_xy, axis=0), axis=1) / np.maximum(dt, 1e-6)


def iter_entries(path: str | Path):
    with open(path, "rb") as f:
        while prefix := f.read(4):
            (size,) = struct.unpack(">L", prefix)
            chunk = f.read(size)
            if len(chunk) != size:  # truncated tail of an interrupted rollout
                return
            yield LogEntry.FromString(chunk)


def _yaw(q) -> float:
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def read_rollout(path: str | Path) -> RolloutLog:
    scene_id = rollout_uuid = ""
    t, xy, yaw, decisions = [], [], [], []
    request_time_us = None
    rig: dict[int, tuple[float, float]] = {}
    for entry in iter_entries(path):
        kind = entry.WhichOneof("log_entry")
        if kind == "driver_request":
            request_time_us = entry.driver_request.time_now_us
        elif kind == "driver_ego_trajectory":
            for p in entry.driver_ego_trajectory.trajectory.poses:
                rig[p.timestamp_us] = (p.pose.vec.x, p.pose.vec.y)
        if kind == "rollout_metadata":
            meta = entry.rollout_metadata.session_metadata
            scene_id, rollout_uuid = meta.scene_id, meta.session_uuid
        elif kind == "actor_poses":
            # The proto documents ego at index 0, but logs do not guarantee it.
            ego = next((a for a in entry.actor_poses.actor_poses if a.actor_id == "EGO"), None)
            if ego is None:
                continue
            p = ego.actor_pose
            t.append(entry.actor_poses.timestamp_us)
            xy.append((p.vec.x, p.vec.y))
            yaw.append(_yaw(p.quat))
        elif kind == "driver_return":
            raw = entry.driver_return.debug_info.unstructured_debug_info
            if not raw:
                continue
            # Written by our own driver process; not untrusted input.
            debug = pickle.loads(raw)
            text = debug.get("reasoning_text")
            if text:
                record = json.loads(text).get(RECORD_KEY)
                if record is not None:
                    record["time_us"] = request_time_us
                    if record["is_decision"]:
                        cands = entry.driver_return.debug_info.sampled_trajectories
                        record["candidates_xy"] = np.array(
                            [[(p.pose.vec.x, p.pose.vec.y) for p in c.poses] for c in cands]
                        )
                        record["candidates_t_us"] = np.array(
                            [p.timestamp_us for p in cands[0].poses], dtype=np.int64
                        )
                    decisions.append(record)
    return RolloutLog(
        path=Path(path),
        scene_id=scene_id,
        rollout_uuid=rollout_uuid,
        ego_t_us=np.asarray(t, dtype=np.int64),
        ego_xy=np.asarray(xy, dtype=float).reshape(-1, 2),
        ego_yaw=np.asarray(yaw, dtype=float),
        rig_t_us=np.array(sorted(rig), dtype=np.int64),
        rig_xy=np.array([rig[k] for k in sorted(rig)], dtype=float).reshape(-1, 2),
        decisions=decisions,
    )


def find_rollouts(run_dir: str | Path) -> list[Path]:
    return sorted(Path(run_dir).glob("**/rollouts/**/*.asl"))
