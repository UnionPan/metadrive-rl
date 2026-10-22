"""Render recorded camera frames of a real NuRec scene and compare to the photos.

Needs the renderer environment (gsplat) and a scene: set METADRIVE_TEST_USDZ to
a ``.usdz`` path. Skipped otherwise.
"""

import io
import os
import zipfile

import numpy as np
import pytest

pytest.importorskip("gsplat")
USDZ = os.environ.get("METADRIVE_TEST_USDZ")
pytestmark = pytest.mark.skipif(not USDZ, reason="METADRIVE_TEST_USDZ not set")

from PIL import Image  # noqa: E402

from metadrive_rl.render.rasterizer import GsplatRenderer, ObjectPose  # noqa: E402
from metadrive_rl.render.scene import NuRecScene  # noqa: E402

CAMERA = "camera_front_wide_120fov"


@pytest.fixture(scope="module")
def renderer(tmp_path_factory):
    cache = os.environ.get("METADRIVE_RENDER_CACHE") or tmp_path_factory.mktemp("cache")
    return GsplatRenderer(NuRecScene(USDZ, cache), "cuda")


def _recorded_frame(scene):
    """The scene's shipped photo for CAMERA and the rig poses it was taken at."""
    with zipfile.ZipFile(scene.usdz_path) as z:
        name = next(n for n in z.namelist() if n.startswith(f"frames/{CAMERA}/"))
        photo = np.asarray(Image.open(io.BytesIO(z.read(name))).convert("RGB"))
    end_us = int(name.rsplit("/", 1)[1].split(".")[0])
    import json

    rig = json.loads(zipfile.ZipFile(scene.usdz_path).read("rig_trajectories.json"))
    traj = rig["rig_trajectories"][0]
    key = next(k for k in traj["cameras_frame_timestamps_us"] if k.startswith(CAMERA))
    frames = traj["cameras_frame_timestamps_us"][key]
    i = next(i for i, (_, b) in enumerate(frames) if b == end_us)
    T_start, T_end = np.array(traj["cameras_frame_T_rig_worlds"][key][i])
    return photo, T_start, T_end, (frames[i][0] + frames[i][1]) // 2


def _objects_at(scene, t_us):
    from scipy.spatial.transform import Rotation, Slerp

    objects = []
    for track_id, (ts, poses) in scene.track_trajectories.items():
        if not ts[0] <= t_us <= ts[-1] or len(ts) < 2:
            continue
        j = int(np.clip(np.searchsorted(ts, t_us) - 1, 0, len(ts) - 2))
        a = (t_us - ts[j]) / max(ts[j + 1] - ts[j], 1)
        T = np.eye(4)
        T[:3, :3] = Slerp([0, 1], Rotation.from_quat(poses[j : j + 2, 3:]))(a).as_matrix()
        T[:3, 3] = (1 - a) * poses[j, :3] + a * poses[j + 1, :3]
        objects.append(ObjectPose(track_id, T))
    return objects


def _psnr(a, b):
    mse = np.mean((a.astype(np.float32) / 255 - b.astype(np.float32) / 255) ** 2)
    return -10 * np.log10(mse)


def test_recorded_frame_matches_photo(renderer):
    scene = renderer.scene
    calib = scene.cameras[CAMERA]
    photo, T_rig_start, T_rig_end, t_mid = _recorded_frame(scene)
    w, h = calib.resolution_wh
    static = renderer.render(calib, w, h, T_rig_start @ calib.T_sensor_rig,
                             T_rig_end @ calib.T_sensor_rig, [])
    full = renderer.render(calib, w, h, T_rig_start @ calib.T_sensor_rig,
                           T_rig_end @ calib.T_sensor_rig, _objects_at(scene, t_mid))
    # Exclude the ego hood (NRE overlays a separate mask; we do not).
    rows = slice(0, int(h * 0.9))
    static, full, photo = static[rows], full[rows], photo[rows]
    vehicle_px = np.abs(full.astype(int) - static.astype(int)).sum(-1) > 15
    on_vehicles = (_psnr(static[vehicle_px], photo[vehicle_px]),
                   _psnr(full[vehicle_px], photo[vehicle_px]))
    print(f"PSNR full frame {_psnr(full, photo):.2f} dB; on vehicle pixels "
          f"({vehicle_px.mean():.1%}) without/with vehicles {on_vehicles[0]:.2f}/{on_vehicles[1]:.2f} dB")
    if out := os.environ.get("METADRIVE_RENDER_DEBUG"):
        Image.fromarray(np.concatenate([photo, full], 0)).save(out)
    assert vehicle_px.mean() > 0.02
    assert on_vehicles[1] > on_vehicles[0] + 3.0  # vehicles land where the photo has them
    assert _psnr(full, photo) > 20.0


def test_reduced_resolution_matches_downsampled(renderer):
    scene = renderer.scene
    calib = scene.cameras[CAMERA]
    _, T_rig_start, T_rig_end, _ = _recorded_frame(scene)
    w, h = calib.resolution_wh
    args = (T_rig_start @ calib.T_sensor_rig, T_rig_end @ calib.T_sensor_rig, [])
    small = renderer.render(calib, 512, 320, *args)
    full = Image.fromarray(renderer.render(calib, w, h, *args)).resize((512, 320), Image.BILINEAR)
    assert small.shape == (320, 512, 3)
    assert _psnr(small, np.asarray(full)) > 24.0
