"""The VaVAM-AIS AlpaGym bundle (M5b) against the trainer and stock VaVAM-B.

Runs only in the AlpaGym environment (``~/ext/alpagym/.venv``, with ``vam`` and
this package installed) on a GPU with the VaVAM assets and the M1 frames.
"""

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("alpagym_runtime")
pytest.importorskip("cosmos_rl")
pytest.importorskip("vam")

from PIL import Image  # noqa: E402

ASSETS = Path.home() / "ext/alpasim/data/drivers/vavam"
BUNDLE = Path("runs/rl/v3_bundle_v0.pt")
FRAMES = sorted(Path("data/episodes/m1/frames").glob("*/*.jpg"))[:12]
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and BUNDLE.exists() and len(FRAMES) >= 12),
    reason="needs GPU, the v3 RL bundle and M1 frames",
)
os.environ.setdefault("VAVAM_MUP_SHAPES_DIR", str(ASSETS / "mup_shapes"))

from alpagym_runtime.inference.types import BatchedModelInput, ModelInput  # noqa: E402
from alpagym_runtime.policies.registry import get_policy_bundle  # noqa: E402
from alpagym_runtime.replay import ActionSelection  # noqa: E402
from transformers import AutoConfig  # noqa: E402

from metadrive_ais.alpagym.inference import VaVAMAISInferenceModel, build_model_inputs  # noqa: E402
from metadrive_ais.alpagym.model import MODEL_TYPE, export_checkpoint  # noqa: E402
from metadrive_ais.models.vavam_loading import load_vavam  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.backends.cudnn.deterministic = True  # the VQ encoder is otherwise run-to-run noisy
    torch.backends.cudnn.benchmark = False
    ckpt = export_checkpoint(BUNDLE, tmp_path_factory.mktemp("ckpt"))
    get_policy_bundle("vavam_ais").install_runtime_bridge()
    from cosmos_rl.policy.model.base import ModelRegistry

    cls = ModelRegistry._MODEL_REGISTRY[MODEL_TYPE]
    cls = cls[0] if isinstance(cls, tuple) else cls
    trainer = cls.from_pretrained(AutoConfig.from_pretrained(ckpt), str(ckpt))
    trainer.load_hf_weights(str(ckpt), device="cuda")
    model = VaVAMAISInferenceModel(ckpt, torch.device("cuda"), {"step_dt_s": 0.5})
    frames = [np.asarray(Image.open(f).convert("RGB")) for f in FRAMES]
    return trainer, model, frames


def _item(frames, start, seed):
    clip = torch.from_numpy(np.stack(frames[start : start + 8])).permute(0, 3, 1, 2)
    route = torch.full((20, 2), float("nan"))
    route[:5] = torch.tensor([[5.0 * i, 0.0] for i in range(1, 6)])
    return ModelInput(
        ego_history_xyz=torch.tensor([[[-4.0, 0.2, 0.0], [0.0, 0.0, 0.0]]]),
        ego_history_rot=torch.eye(3).repeat(1, 2, 1, 1),
        camera_frames=clip,
        camera_indices=torch.zeros(8, dtype=torch.int64),
        relative_timestamps=torch.zeros(8, dtype=torch.int64),
        route_xy=route,
        seed=torch.tensor(seed),
    )


def test_rollout_log_prob_equals_trainer_recomputation(setup):
    trainer, model, frames = setup
    items = [_item(frames, 0, 11), _item(frames, 4, 12)]
    out = model.sample_trajectories_from_data(BatchedModelInput.stack(items), None)
    assert out.pred_xyz.shape == (2, 1, 1, 6, 3) and out.pred_rot.shape == (2, 1, 1, 6, 3, 3)
    for mi, mo in zip(items, out.unbind(), strict=True):
        replay = model.build_policy_replay_data(mi, mo, ActionSelection(0, 0))
        inputs, old = build_model_inputs(replay)
        new = trainer(**{k: v[None].cuda() for k, v in inputs.items()}, return_log_prob=True)
        assert float(new["log_probs"]) == pytest.approx(float(old), abs=1e-5)


def test_seeded_latents_are_reproducible(setup):
    _, model, frames = setup
    batch = BatchedModelInput.stack([_item(frames, 0, 21)])
    a = model.sample_trajectories_from_data(batch, None).extra["c"]
    b = model.sample_trajectories_from_data(batch, None).extra["c"]
    assert torch.equal(a, b)


def test_without_latent_residual_the_policy_is_stock_vavam(setup):
    """Same tokens, command and flow seed: stock VaVAM-B, bit for bit.

    VaVAM's TorchScript VQ encoder re-profiles on its first call per batch size,
    so tokens depend on call history and batch size at the ~0.3% level (AlpaSim's
    driver behaves the same). The test therefore feeds stock VaVAM the tokens the
    AIS path actually used.
    """
    _, model, frames = setup
    weight = model.latent.decoder.proj.weight.data.clone()
    model.latent.decoder.proj.weight.data.zero_()  # the v0 adapter gate is already 0
    seen = {}
    forward = model.vam.forward

    def spy(tokens, command, dtype):
        seen.update(tokens=tokens.clone(), command=command.clone())
        return forward(tokens, command, dtype)

    model.vam.forward = spy
    try:
        batch = BatchedModelInput.stack([_item(frames, 0, 1), _item(frames, 4, 2)])
        torch.manual_seed(0)
        ais = model.sample_trajectories_from_data(batch, None).pred_xyz[:, 0, 0, :, :2]
    finally:
        model.vam.forward = forward
        model.latent.decoder.proj.weight.data.copy_(weight)
    stock = load_vavam(device=torch.device("cuda"))
    torch.manual_seed(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        ref = stock(seen["tokens"], seen["command"], torch.float16)
    assert seen["command"].tolist() == [[2], [2]]
    assert torch.equal(ais.cpu(), ref.float().reshape(2, 6, 2).cpu())
