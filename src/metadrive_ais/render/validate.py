"""Compare renders with the camera photos shipped inside NuRec scenes.

Each scene ships one recorded photo per camera with its rig poses, which gives
a ground-truth check of geometry, camera model and actor placement.

Usage: python -m metadrive_ais.render.validate <usdz>... [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .rasterizer import GsplatRenderer
from .recording import recorded_frame, recorded_objects
from .scene import CameraCalibration, NuRecScene

HOOD_FRACTION = 0.1  # bottom band excluded: NRE overlays an ego-hood mask, we do not


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) / 255 - b.astype(np.float32) / 255) ** 2)
    return float(-10 * np.log10(mse))


def compare(renderer: GsplatRenderer, camera: str) -> tuple[dict, np.ndarray]:
    scene = renderer.scene
    calib = scene.cameras[camera]
    frame = recorded_frame(scene, camera)
    w, h = calib.resolution_wh
    poses = (
        frame.T_world_rig_start @ calib.T_sensor_rig,
        frame.T_world_rig_end @ calib.T_sensor_rig,
    )
    static = renderer.render(calib, w, h, *poses, [])
    full = renderer.render(calib, w, h, *poses, recorded_objects(scene, frame.mid_time_us))
    rows = slice(0, int(h * (1 - HOOD_FRACTION)))
    photo, static, full = frame.photo[rows], static[rows], full[rows]
    vehicle_px = np.abs(full.astype(int) - static.astype(int)).sum(-1) > 15
    stats = {
        "scene_id": scene.scene_id,
        "psnr_full_db": psnr(full, photo),
        "vehicle_px_fraction": float(vehicle_px.mean()),
        "psnr_vehicle_px_without_db": psnr(static[vehicle_px], photo[vehicle_px])
        if vehicle_px.any()
        else None,
        "psnr_vehicle_px_with_db": psnr(full[vehicle_px], photo[vehicle_px])
        if vehicle_px.any()
        else None,
    }
    return stats, np.concatenate([photo, full], 0)


# The synthetic front camera of AlpaSim's VaVAM preset (vavam_configs.yaml).
VAVAM_PINHOLE = {
    "resolution_wh": (1920, 1080),
    "focal_length": (1545.0, 1545.0),
    "principal_point": (960.0, 560.0),
    "radial_coeffs": (-0.356123, 0.172545, -0.05231, 0.0, 0.0, 0.0),
    "tangential_coeffs": (-0.00213, 0.000464),
}


def opencv_project(dirs: np.ndarray, calib: CameraCalibration) -> np.ndarray:
    """Pixels of camera-frame directions (z forward) under OpenCV's pinhole model."""
    x, y = dirs[:, 0] / dirs[:, 2], dirs[:, 1] / dirs[:, 2]
    k = np.pad(np.asarray(calib.radial_coeffs, float), (0, 6))[:6]
    p1, p2 = np.pad(np.asarray(calib.tangential_coeffs, float), (0, 2))[:2]
    s = np.pad(np.asarray(calib.thin_prism_coeffs, float), (0, 4))[:4]
    r2 = x * x + y * y
    radial = (1 + k[0] * r2 + k[1] * r2**2 + k[2] * r2**3) / (
        1 + k[3] * r2 + k[4] * r2**2 + k[5] * r2**3
    )
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s[0] * r2 + s[1] * r2**2
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s[2] * r2 + s[3] * r2**2
    (fx, fy), (cx, cy) = calib.focal_length, calib.principal_point
    return np.stack([fx * xd + cx, fy * yd + cy], 1)


def ftheta_project(dirs: np.ndarray, calib: CameraCalibration) -> np.ndarray:
    """Pixels of camera-frame directions under the NuRec f-theta model."""
    norm_xy = np.linalg.norm(dirs[:, :2], axis=1)
    theta = np.arctan2(norm_xy, dirs[:, 2])
    r = np.polynomial.polynomial.polyval(theta, calib.angle_to_pixeldist_poly)
    nx, ny = dirs[:, 0] / norm_xy, dirs[:, 1] / norm_xy
    c, d, e = calib.linear_cde
    cx, cy = calib.principal_point
    return np.stack([cx + c * r * nx + d * r * ny, cy + e * r * nx + r * ny], 1)


def sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear lookup of pixel-centre coordinates (u right, v down)."""
    h, w = image.shape[:2]
    u, v = uv[:, 0] - 0.5, uv[:, 1] - 0.5
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    du, dv = (u - u0)[:, None], (v - v0)[:, None]
    u0, v0 = np.clip(u0, 0, w - 2), np.clip(v0, 0, h - 2)
    img = image.astype(np.float32)
    top = img[v0, u0] * (1 - du) + img[v0, u0 + 1] * du
    bottom = img[v0 + 1, u0] * (1 - du) + img[v0 + 1, u0 + 1] * du
    return top * (1 - dv) + bottom * dv


def pinhole_consistency(renderer: GsplatRenderer, camera: str = "camera_front_wide_120fov") -> dict:
    """Check the pinhole path against OpenCV's model and the validated f-theta path.

    Renders VaVAM's distorted pinhole camera, the same camera without
    distortion, and the scene's f-theta camera, all at the f-theta camera's
    recorded pose. Viewing directions on the undistorted pixel grid are then
    projected into each image with independent NumPy models and the sampled
    colours compared (PSNR, dB). A mapping that ignores the distortion is the
    negative control.
    """
    scene = renderer.scene
    ftheta = scene.cameras[camera]
    frame = recorded_frame(scene, camera)
    poses = (
        frame.T_world_rig_start @ ftheta.T_sensor_rig,
        frame.T_world_rig_end @ ftheta.T_sensor_rig,
    )
    objects = recorded_objects(scene, frame.mid_time_us)
    distorted = CameraCalibration(
        logical_id=camera,
        T_sensor_rig=ftheta.T_sensor_rig,
        shutter_type=ftheta.shutter_type,
        camera_model="pinhole",
        **VAVAM_PINHOLE,
    )
    ideal = CameraCalibration(
        **{**distorted.__dict__, "radial_coeffs": (), "tangential_coeffs": ()}
    )
    w, h = distorted.resolution_wh
    img_d = renderer.render(distorted, w, h, *poses, objects)
    img_0 = renderer.render(ideal, w, h, *poses, objects)
    img_f = renderer.render(ftheta, *ftheta.resolution_wh, *poses, objects)

    (fx, fy), (cx, cy) = ideal.focal_length, ideal.principal_point
    u, v = np.meshgrid(np.arange(8, w - 8, 4) + 0.5, np.arange(8, int(h * 0.8), 4) + 0.5)
    uv0 = np.stack([u.ravel(), v.ravel()], 1)
    dirs = np.stack([(uv0[:, 0] - cx) / fx, (uv0[:, 1] - cy) / fy, np.ones(len(uv0))], 1)
    uv_d, uv_f = opencv_project(dirs, distorted), ftheta_project(dirs, ftheta)
    fw, fh = ftheta.resolution_wh
    keep = (
        (uv_d[:, 0] > 2)
        & (uv_d[:, 0] < w - 2)
        & (uv_d[:, 1] > 2)
        & (uv_d[:, 1] < h - 2)
        & (uv_f[:, 0] > 2)
        & (uv_f[:, 0] < fw - 2)
        & (uv_f[:, 1] > 2)
        & (uv_f[:, 1] < fh - 2)
    )
    ref = sample(img_0, uv0[keep])

    def db(other: np.ndarray) -> float:
        mse = np.mean((ref / 255 - other / 255) ** 2)
        return float(-10 * np.log10(mse))

    return {
        "scene_id": scene.scene_id,
        "samples": int(keep.sum()),
        "distortion_vs_opencv_db": db(sample(img_d, uv_d[keep])),
        "distortion_ignored_db (control)": db(sample(img_d, uv0[keep])),
        "pinhole_vs_ftheta_db": db(sample(img_f, uv_f[keep])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("usdz", nargs="+", type=Path)
    parser.add_argument("--camera", default="camera_front_wide_120fov")
    parser.add_argument(
        "--out", type=Path, help="write a JSON summary and side-by-side images here"
    )
    parser.add_argument("--max-scale-m", type=float, help="prune larger static Gaussians")
    parser.add_argument("--min-opacity", type=float, default=0.0)
    args = parser.parse_args()
    results = []
    for path in args.usdz:
        renderer = GsplatRenderer(NuRecScene(path), "cuda", args.max_scale_m, args.min_opacity)
        stats, pair = compare(renderer, args.camera)
        results.append(stats)
        print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}))
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pair).resize((pair.shape[1] // 4, pair.shape[0] // 4)).save(
                args.out / f"{stats['scene_id']}.jpg", quality=85
            )
        del renderer
    if args.out:
        (args.out / "summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
