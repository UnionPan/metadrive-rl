"""Render NuRec scenes with gsplat's 3DGUT rasterizer (supports Turing GPUs).

Cameras use the scene's f-theta models; rendering at a resolution other than
the native one scales pixel coordinates per axis. Sky pixels (where the
Gaussians leave transmittance) are filled with a constant colour, see
:attr:`NuRecScene.sky_color`.
"""

from __future__ import annotations

from dataclasses import dataclass

import gsplat
import numpy as np
import torch
from gsplat.cuda._wrapper import FThetaCameraDistortionParameters, FThetaPolynomialType

from .scene import CameraCalibration, GaussianSet, NuRecScene

_SHUTTERS = {
    "ROLLING_TOP_TO_BOTTOM": gsplat.RollingShutterType.ROLLING_TOP_TO_BOTTOM,
    "ROLLING_LEFT_TO_RIGHT": gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT,
    "ROLLING_BOTTOM_TO_TOP": gsplat.RollingShutterType.ROLLING_BOTTOM_TO_TOP,
    "ROLLING_RIGHT_TO_LEFT": gsplat.RollingShutterType.ROLLING_RIGHT_TO_LEFT,
    "GLOBAL": gsplat.RollingShutterType.GLOBAL,
}


@dataclass(frozen=True)
class ObjectPose:
    track_id: str
    T_world_object: np.ndarray  # (4, 4) cuboid pose in the scene's world frame


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def _pixel_scale(calib: CameraCalibration, width: int, height: int) -> tuple[float, float]:
    native_w, native_h = calib.resolution_wh
    return width / native_w, height / native_h


def _ftheta(calib: CameraCalibration) -> FThetaCameraDistortionParameters:
    return FThetaCameraDistortionParameters(
        reference_poly=FThetaPolynomialType.PIXELDIST_TO_ANGLE,
        pixeldist_to_angle_poly=calib.pixeldist_to_angle_poly,
        angle_to_pixeldist_poly=calib.angle_to_pixeldist_poly,
        max_angle=calib.max_angle,
        linear_cde=calib.linear_cde,
    )


class GsplatRenderer:
    def __init__(self, scene: NuRecScene, device: torch.device | str = "cuda"):
        self.scene = scene.to(device)
        self.device = torch.device(device)

    def _placed_objects(self, objects: list[ObjectPose]) -> list[GaussianSet]:
        """Move each requested actor's Gaussians from cuboid frame to the NRE frame."""
        from scipy.spatial.transform import Rotation

        known = [o for o in objects if o.track_id in self.scene.track_slices]
        if not known:  # actors without a reconstruction (e.g. filtered tracks) are skipped
            return []
        T = self.scene.world_to_nre @ np.stack([o.T_world_object for o in known])  # (K, 4, 4)
        xyzw = Rotation.from_matrix(T[:, :3, :3]).as_quat()
        dev = self.device
        R = torch.tensor(T[:, :3, :3], dtype=torch.float32, device=dev)
        t = torch.tensor(T[:, :3, 3], dtype=torch.float32, device=dev)
        q = torch.tensor(xyzw[:, [3, 0, 1, 2]], dtype=torch.float32, device=dev)
        bounds = [self.scene.track_slices[o.track_id] for o in known]
        counts = torch.tensor([b - a for a, b in bounds], device=dev)
        index = torch.cat([torch.arange(a, b, device=dev) for a, b in bounds])
        owner = torch.repeat_interleave(torch.arange(len(known), device=dev), counts)
        d = self.scene.dynamic
        means = torch.einsum("nij,nj->ni", R[owner], d.means[index]) + t[owner]
        return [GaussianSet(means, _quat_mul(q[owner], d.quats[index]), d.scales[index],
                            d.opacities[index], d.sh[index])]

    def _viewmat(self, T_world_camera: np.ndarray) -> torch.Tensor:
        T = self.scene.world_to_nre @ T_world_camera
        return torch.tensor(np.linalg.inv(T), dtype=torch.float32, device=self.device)[None]

    @torch.no_grad()
    def render(
        self,
        calib: CameraCalibration,
        width: int,
        height: int,
        T_world_camera_start: np.ndarray,
        T_world_camera_end: np.ndarray,
        objects: list[ObjectPose],
        shutter_type: str | None = None,
    ) -> np.ndarray:
        """Render one frame; returns (H, W, 3) uint8."""
        g = GaussianSet.concat([self.scene.static, *self._placed_objects(objects)])
        sx, sy = _pixel_scale(calib, width, height)
        cx, cy = calib.principal_point
        K = torch.tensor([[[sx, 0.0, cx * sx], [0.0, sy, cy * sy], [0.0, 0.0, 1.0]]],
                         device=self.device)
        colors, alpha, _ = gsplat.rasterization(
            g.means, g.quats, g.scales, g.opacities, g.sh,
            self._viewmat(T_world_camera_start), K, width, height,
            sh_degree=3, camera_model="ftheta", ftheta_coeffs=_ftheta(calib),
            with_ut=True, with_eval3d=True, packed=False,
            rolling_shutter=_SHUTTERS[shutter_type or calib.shutter_type],
            viewmats_rs=self._viewmat(T_world_camera_end),
            near_plane=0.1, far_plane=1e4,
        )
        image = colors[0] + (1.0 - alpha[0]) * self.scene.sky_color
        return (image.clamp(0, 1) * 255).round().byte().cpu().numpy()
