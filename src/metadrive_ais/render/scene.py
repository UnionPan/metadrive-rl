"""NuRec scene (``.usdz``) loader for the gsplat renderer.

A NuRec scene stores a 3D-Gaussian reconstruction of a recorded drive:

* ``background`` and ``road`` layers: static Gaussians in the NRE frame.
* ``dynamic_rigids``: vehicle Gaussians in each cuboid's local frame, tagged
  with a cuboid index. Cuboid ``i`` is the track whose recorded time span
  equals ``time_embed.timestamps_us_ranges[i]``.
* Per-camera background textures, camera calibrations, and the rig trajectory.

Conventions (verified against the recorded camera frames shipped in the
scene): scales are log-space, densities are logits, rotations are ``wxyz``
quaternions, colour is degree-3 SH with the DC term in ``features_albedo[..., 0]``
and the remaining 15 coefficients in ``features_specular`` laid out (15, 3).
Rig and track poses are in the scene's world frame; ``world_to_nre`` maps
them into the frame of the Gaussians.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_state_dict

LAYER_PREFIX = "model.gaussians_nodes."


@dataclass
class GaussianSet:
    means: torch.Tensor  # (N, 3)
    quats: torch.Tensor  # (N, 4) wxyz, normalised
    scales: torch.Tensor  # (N, 3) linear
    opacities: torch.Tensor  # (N,)
    sh: torch.Tensor  # (N, 16, 3)

    def __len__(self) -> int:
        return self.means.shape[0]

    def to(self, device: torch.device | str) -> GaussianSet:
        return GaussianSet(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))

    @staticmethod
    def concat(sets: list[GaussianSet]) -> GaussianSet:
        return GaussianSet(
            *(torch.cat([getattr(s, f) for s in sets]) for f in GaussianSet.__dataclass_fields__)
        )

    def slice(self, start: int, stop: int) -> GaussianSet:
        return GaussianSet(*(getattr(self, f)[start:stop] for f in self.__dataclass_fields__))


@dataclass
class CameraCalibration:
    logical_id: str
    T_sensor_rig: np.ndarray  # (4, 4) camera pose in the rig frame (OpenCV axes)
    resolution_wh: tuple[int, int]
    principal_point: tuple[float, float]
    shutter_type: str
    camera_model: str = "ftheta"  # "ftheta" (NuRec rig cameras) or "pinhole" (OpenCV)
    # f-theta
    pixeldist_to_angle_poly: tuple[float, ...] = ()
    angle_to_pixeldist_poly: tuple[float, ...] = ()
    max_angle: float = 0.0
    linear_cde: tuple[float, float, float] = (1.0, 0.0, 0.0)
    # OpenCV pinhole: focal lengths in pixels and distortion in normalised
    # image coordinates (k1..k6, p1 p2, s1..s4), as in cv2.projectPoints.
    focal_length: tuple[float, float] = (0.0, 0.0)
    radial_coeffs: tuple[float, ...] = ()
    tangential_coeffs: tuple[float, ...] = ()
    thin_prism_coeffs: tuple[float, ...] = ()


def _gaussians(sd: dict, layer: str, dc: torch.Tensor) -> GaussianSet:
    p = f"{LAYER_PREFIX}{layer}."
    n = sd[p + "positions"].shape[0]
    return GaussianSet(
        means=sd[p + "positions"].float(),
        quats=torch.nn.functional.normalize(sd[p + "rotations"].float(), dim=-1),
        scales=torch.exp(sd[p + "scales"].float()),
        opacities=torch.sigmoid(sd[p + "densities"].float()[:, 0]),
        sh=torch.cat([dc.reshape(n, 1, 3), sd[p + "features_specular"].float().view(n, 15, 3)], 1),
    )


def _dc(sd: dict, layer: str) -> torch.Tensor:
    albedo = sd[f"{LAYER_PREFIX}{layer}.features_albedo"].float()
    # Layers with temporal appearance store extra albedo slots; slot 0 is the
    # static DC term.
    return albedo[:, 0] if albedo.dim() == 3 else albedo


class NuRecScene:
    def __init__(self, usdz_path: str | Path):
        self.usdz_path = Path(usdz_path)
        with zipfile.ZipFile(self.usdz_path) as z:
            rig = json.loads(z.read("rig_trajectories.json"))
            tracks_json = json.loads(z.read("sequence_tracks.json"))
            # The checkpoint is stored uncompressed; reading it into memory in
            # one pass (~3 s) avoids an extracted copy on disk.
            sd = load_state_dict(io.BytesIO(z.read("checkpoint.ckpt")))

        trajectory = rig["rig_trajectories"][0]
        self.scene_id: str = trajectory["sequence_id"]
        self.world_to_nre = np.array(rig["world_to_nre"]["matrix"], dtype=np.float64)
        self.rig_timestamps_us = np.array(trajectory["T_rig_world_timestamps_us"], dtype=np.int64)
        self.rig_poses = np.array(trajectory["T_rig_worlds"], dtype=np.float64)
        self.cameras = {
            cal["logical_sensor_name"]: CameraCalibration(
                logical_id=cal["logical_sensor_name"],
                T_sensor_rig=np.array(cal["T_sensor_rig"], dtype=np.float64),
                resolution_wh=tuple(cal["camera_model"]["parameters"]["resolution"]),
                principal_point=tuple(cal["camera_model"]["parameters"]["principal_point"]),
                pixeldist_to_angle_poly=tuple(
                    cal["camera_model"]["parameters"]["pixeldist_to_angle_poly"]
                ),
                angle_to_pixeldist_poly=tuple(
                    cal["camera_model"]["parameters"]["angle_to_pixeldist_poly"]
                ),
                max_angle=float(cal["camera_model"]["parameters"]["max_angle"]),
                linear_cde=tuple(cal["camera_model"]["parameters"]["linear_cde"]),
                shutter_type=cal["camera_model"]["parameters"]["shutter_type"],
            )
            for cal in rig["camera_calibrations"].values()
            if cal["camera_model"]["type"] == "ftheta"
        }

        self.static = GaussianSet.concat(
            [_gaussians(sd, layer, _dc(sd, layer)) for layer in ("background", "road")]
        )
        # ``model.background.textures`` holds one sky/background panorama per
        # camera; its parameterisation is undocumented and no mapping we tried
        # beat a flat colour against recorded frames, so the sky is rendered as
        # the mean colour of the textures' upper (sky) half.
        textures = sd["model.background.textures"][0].float()  # (n_cameras, R, R, 3)
        self.sky_color = textures[:, : textures.shape[1] // 2].mean(dim=(0, 1, 2))
        self._load_dynamic(sd, tracks_json)

    def _load_dynamic(self, sd: dict, tracks_json: dict) -> None:
        layer = "dynamic_rigids"
        p = f"{LAYER_PREFIX}{layer}."
        tracks = next(iter(tracks_json.values()))["tracks_data"]
        span_to_track = {
            (ts[0], ts[-1]): track_id
            for track_id, ts in zip(
                tracks["tracks_id"], tracks["tracks_timestamps_us"], strict=True
            )
        }
        ranges = sd[p + "time_embed.timestamps_us_ranges"].tolist()
        cuboid_track_ids = [span_to_track[(int(a), int(b))] for a, b in ranges]

        dynamic = _gaussians(sd, layer, _dc(sd, layer))
        cuboid = sd[p + "gaussian_cuboid_ids"].long()
        order = torch.argsort(cuboid, stable=True)
        dynamic = GaussianSet(
            *(getattr(dynamic, f)[order] for f in GaussianSet.__dataclass_fields__)
        )
        counts = torch.bincount(cuboid, minlength=len(cuboid_track_ids)).tolist()
        bounds = np.concatenate([[0], np.cumsum(counts)]).astype(int)
        self.dynamic = dynamic
        # track id -> (start, stop) slice into ``self.dynamic``
        self.track_slices = {
            track_id: (int(bounds[i]), int(bounds[i + 1]))
            for i, track_id in enumerate(cuboid_track_ids)
            if bounds[i + 1] > bounds[i]
        }
        self.track_trajectories = {
            track_id: (np.array(ts, dtype=np.int64), np.array(poses, dtype=np.float64))
            for track_id, ts, poses in zip(
                tracks["tracks_id"],
                tracks["tracks_timestamps_us"],
                tracks["tracks_poses"],
                strict=True,
            )
        }

    def to(self, device: torch.device | str) -> NuRecScene:
        self.static = self.static.to(device)
        self.dynamic = self.dynamic.to(device)
        self.sky_color = self.sky_color.to(device)
        return self

    def rig_pose_at(self, t_us: int) -> np.ndarray:
        """Recorded rig pose (world frame) at ``t_us``, linearly interpolated."""
        from scipy.spatial.transform import Rotation, Slerp

        ts = self.rig_timestamps_us
        j = int(np.clip(np.searchsorted(ts, t_us) - 1, 0, len(ts) - 2))
        a = float(np.clip((t_us - ts[j]) / max(ts[j + 1] - ts[j], 1), 0.0, 1.0))
        rots = Rotation.from_matrix(self.rig_poses[j : j + 2, :3, :3])
        T = np.eye(4)
        T[:3, :3] = Slerp([0, 1], rots)(a).as_matrix()
        T[:3, 3] = (1 - a) * self.rig_poses[j, :3, 3] + a * self.rig_poses[j + 1, :3, 3]
        return T
