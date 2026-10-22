"""Compare renders with the camera photos shipped inside NuRec scenes.

Each scene ships one recorded photo per camera with its rig poses, which gives
a ground-truth check of geometry, camera model and actor placement.

Usage: python -m metadrive_rl.render.validate <usdz>... [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .rasterizer import GsplatRenderer
from .recording import recorded_frame, recorded_objects
from .scene import NuRecScene

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
