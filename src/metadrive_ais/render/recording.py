"""Recorded reference data shipped inside NuRec scenes.

Each scene contains one recorded frame per camera (``frames/<camera>/<t>.jpeg``)
with the rig poses it was taken at, and the recorded actor tracks.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from .rasterizer import ObjectPose
from .scene import NuRecScene


def cameras_with_photos(scene: NuRecScene) -> list[str]:
    with zipfile.ZipFile(scene.usdz_path) as z:
        names = {
            n.split("/")[1] for n in z.namelist() if n.startswith("frames/") and n.count("/") == 2
        }
    return sorted(c for c in names if c in scene.cameras)


@dataclass
class RecordedFrame:
    photo: np.ndarray  # (H, W, 3) uint8
    T_world_rig_start: np.ndarray
    T_world_rig_end: np.ndarray
    mid_time_us: int


def recorded_frame(scene: NuRecScene, camera: str) -> RecordedFrame:
    with zipfile.ZipFile(scene.usdz_path) as z:
        name = next(n for n in z.namelist() if n.startswith(f"frames/{camera}/"))
        photo = np.asarray(Image.open(io.BytesIO(z.read(name))).convert("RGB"))
        traj = json.loads(z.read("rig_trajectories.json"))["rig_trajectories"][0]
    end_us = int(name.rsplit("/", 1)[1].split(".")[0])
    key = next(k for k in traj["cameras_frame_timestamps_us"] if k.startswith(camera))
    frames = traj["cameras_frame_timestamps_us"][key]
    i = next(i for i, (_, b) in enumerate(frames) if b == end_us)
    start, end = np.array(traj["cameras_frame_T_rig_worlds"][key][i])
    return RecordedFrame(photo, start, end, (frames[i][0] + frames[i][1]) // 2)


def recorded_objects(scene: NuRecScene, t_us: int) -> list[ObjectPose]:
    """Actors at their recorded poses (linear/slerp interpolation) at ``t_us``."""
    objects = []
    for track_id, (ts, poses) in scene.track_trajectories.items():
        if len(ts) < 2 or not ts[0] <= t_us <= ts[-1]:
            continue
        j = int(np.clip(np.searchsorted(ts, t_us) - 1, 0, len(ts) - 2))
        a = (t_us - ts[j]) / max(ts[j + 1] - ts[j], 1)
        T = np.eye(4)
        T[:3, :3] = Slerp([0, 1], Rotation.from_quat(poses[j : j + 2, 3:]))(a).as_matrix()
        T[:3, 3] = (1 - a) * poses[j, :3] + a * poses[j + 1, :3]
        objects.append(ObjectPose(track_id, T))
    return objects
