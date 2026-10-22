"""Render recorded camera frames of a real NuRec scene and compare to the photos.

Needs the renderer environment (gsplat) and a scene: set METADRIVE_TEST_USDZ to
a ``.usdz`` path. Skipped otherwise.
"""

import os

import numpy as np
import pytest

pytest.importorskip("gsplat")
USDZ = os.environ.get("METADRIVE_TEST_USDZ")
pytestmark = pytest.mark.skipif(not USDZ, reason="METADRIVE_TEST_USDZ not set")

from PIL import Image  # noqa: E402

from metadrive_rl.render.rasterizer import GsplatRenderer  # noqa: E402
from metadrive_rl.render.scene import NuRecScene  # noqa: E402
from metadrive_rl.render.validate import compare, psnr, recorded_frame  # noqa: E402

CAMERA = "camera_front_wide_120fov"


@pytest.fixture(scope="module")
def renderer():
    return GsplatRenderer(NuRecScene(USDZ), "cuda")


def test_recorded_frame_matches_photo(renderer):
    stats, pair = compare(renderer, CAMERA)
    print(stats)
    if out := os.environ.get("METADRIVE_RENDER_DEBUG"):
        Image.fromarray(pair).save(out)
    assert stats["vehicle_px_fraction"] > 0.02
    # Vehicles land where the photo has them.
    assert stats["psnr_vehicle_px_with_db"] > stats["psnr_vehicle_px_without_db"] + 3.0
    assert stats["psnr_full_db"] > 20.0


def test_reduced_resolution_matches_downsampled(renderer):
    calib = renderer.scene.cameras[CAMERA]
    frame = recorded_frame(renderer.scene, CAMERA)
    w, h = calib.resolution_wh
    args = (
        frame.T_world_rig_start @ calib.T_sensor_rig,
        frame.T_world_rig_end @ calib.T_sensor_rig,
        [],
    )
    small = renderer.render(calib, 512, 320, *args)
    full = Image.fromarray(renderer.render(calib, w, h, *args)).resize((512, 320), Image.BILINEAR)
    assert small.shape == (320, 512, 3)
    assert psnr(small, np.asarray(full)) > 24.0
