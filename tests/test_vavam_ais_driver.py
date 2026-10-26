"""The VaVAM-AIS AlpaSim driver against the stock VaVAM driver.

Needs a GPU, the AlpaSim venv, the VaVAM-B assets, DINOv3-S access and the M1
episode frames (``data/episodes/m1``); skipped otherwise.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
base = pytest.importorskip("alpasim_driver.models.base")
vam_model = pytest.importorskip("alpasim_driver.models.vam_model")
from alpasim_grpc.v0.common_pb2 import PoseAtTime  # noqa: E402
from PIL import Image  # noqa: E402

ASSETS = Path(os.environ.get("VAVAM_ASSETS", Path.home() / "ext/alpasim/data/drivers/vavam"))
CKPT, TOKENIZER = (
    ASSETS / "VAM_width_1024_pretrained_139k.pt",
    ASSETS / "VQ_ds16_16384_llamagen_encoder.jit",
)
FRAMES = sorted(Path("data/episodes/m1/frames").glob("*/*.jpg"))[:12]
pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and torch.cuda.is_available() and len(FRAMES) >= 12),
    reason="needs GPU, VaVAM-B assets and M1 frames",
)

from metadrive_ais.drivers.vavam_ais import VaVAMAISModel, save_bundle  # noqa: E402
from metadrive_ais.memory.ais import MemoryEncoder  # noqa: E402
from metadrive_ais.models.latent_policy import LatentChannel  # noqa: E402
from metadrive_ais.models.vavam_conditioning import GatedAdapter  # noqa: E402

CAM = "camera_front_wide_120fov"


def _input(frames, t_index, session):
    poses = []
    for k in range(t_index + 1):
        p = PoseAtTime(timestamp_us=k * 500_000)
        p.pose.vec.x, p.pose.quat.w = 4.0 * k, 1.0
        poses.append(p)
    images = [np.asarray(Image.open(f).convert("RGB")) for f in frames]
    return base.PredictionInput(
        camera_images={CAM: [(k, im) for k, im in enumerate(images)]},
        command=base.DriveCommand.STRAIGHT,
        speed=8.0,
        acceleration=0.0,
        ego_pose_history=poses,
        inference_seed=t_index,
        previous_plan=None,
        route=None,
        session_uuid=session,
    )


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    device = torch.device("cuda")
    with pytest.MonkeyPatch.context() as mp:
        # Both models load VaVAM with its muP base shapes (patches/alpasim/0005).
        mp.setenv("VAVAM_MUP_SHAPES_DIR", str(ASSETS / "mup_shapes"))
        stock = vam_model.VAMModel(str(CKPT), str(TOKENIZER), device, [CAM], context_length=1)
        torch.manual_seed(0)
        memory, adapter = MemoryEncoder(768, "gru"), GatedAdapter(256)
        bundle = tmp_path_factory.mktemp("bundle") / "zero_gate.pt"
        save_bundle(bundle, CKPT, "dinov3-small", memory, adapter)
        loaded = torch.load(bundle, map_location="cpu", weights_only=True)
        ais = VaVAMAISModel(loaded, str(TOKENIZER), device, [CAM], context_length=1)
    return stock, ais


def _run(model, inputs, seed=0):
    """Waypoint positions (K, T, 3), rig frame, of each prediction."""
    torch.manual_seed(seed)
    out = [p.candidate_positions for p in model.predict_batch(inputs)]
    assert all(a.ndim == 3 and a.shape[1] >= 6 and np.abs(a).max() > 1.0 for a in out)
    return out


def test_zero_gate_driver_matches_stock_vavam(models):
    stock, ais = models
    inputs = [_input([FRAMES[i]], i, "s-parity") for i in range(3)]
    for inp in inputs:
        a, b = _run(stock, [inp]), _run(ais, [inp])
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_sessions_are_isolated_and_memory_matters(models):
    _, ais = models
    with torch.no_grad():
        ais._adapter.gate.fill_(1.0)
    try:
        alone = [_run(ais, [_input([FRAMES[i]], i, "s-alone")])[0] for i in range(4)]
        mixed = []
        for i in range(4):
            mixed.append(_run(ais, [_input([FRAMES[i]], i, "s-a")])[0])
            _run(ais, [_input([FRAMES[8 + i]], i, "s-b")])  # another session in between
        assert all(np.array_equal(x, y) for x, y in zip(alone, mixed, strict=True))
        # Same frame, different history: the memory changes the trajectory.
        fresh = _run(ais, [_input([FRAMES[3]], 0, "s-fresh")])[0]
        assert not np.array_equal(fresh, alone[3])
    finally:
        with torch.no_grad():
            ais._adapter.gate.zero_()


def test_latent_draws_are_reproducible_and_logged(models, tmp_path):
    """Seeded per session and query; the logged log q is recomputable from (z, c)."""
    _, ais = models
    torch.manual_seed(0)
    latent = LatentChannel(256, latent_dim=4).to(ais._device).eval()
    with torch.no_grad():
        latent.decoder.proj.weight.mul_(20.0)  # make the latent visibly steer
    log = tmp_path / "latent.jsonl"
    ais._latent, ais._latent_log, ais._latent_mode = latent, str(log), "sample"
    try:
        first = [_run(ais, [_input([FRAMES[i]], i, "s-latent")])[0] for i in range(3)]
        del ais._sessions["s-latent"]
        again = [_run(ais, [_input([FRAMES[i]], i, "s-latent")])[0] for i in range(3)]
        assert all(np.array_equal(x, y) for x, y in zip(first, again, strict=True))
        other = _run(ais, [_input([FRAMES[0]], 0, "s-other")])[0]
        assert not np.array_equal(other, first[0])  # same frame, different latent draw
        records = [json.loads(line) for line in log.read_text().splitlines()]
        assert [r["query"] for r in records[:3]] == [0, 1, 2]
        assert records[0]["c"] == records[3]["c"] and records[0]["c"] != records[1]["c"]
        for r in records:
            dist = latent.prior(torch.tensor(r["z"], device=ais._device))
            logp = dist.log_prob(torch.tensor(r["c"], device=ais._device)).sum()
            assert abs(float(logp) - r["logp"]) < 1e-4
    finally:
        ais._latent, ais._latent_log = None, None
