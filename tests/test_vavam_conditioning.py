"""Zero-gate parity and controllability of the VaVAM-B conditioning adapter.

Needs a GPU, the AlpaSim venv (``vam``, ``alpasim_driver``) and the VaVAM-B
checkpoint (``data/download_vavam_assets.sh --model VaVAM-B``); skipped otherwise.
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vam")
vam_model = pytest.importorskip("alpasim_driver.models.vam_model")

CKPT = Path(
    os.environ.get(
        "VAVAM_CKPT",
        Path.home() / "ext/alpasim/data/drivers/vavam/VAM_width_1024_pretrained_139k.pt",
    )
)
pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and torch.cuda.is_available()), reason="needs GPU and VaVAM-B checkpoint"
)

from metadrive_ais.models.vavam_conditioning import (  # noqa: E402
    GatedAdapter,
    attach_conditioning,
)


@pytest.fixture(scope="module")
def vam():
    return vam_model.load_inference_VAM(str(CKPT), "cuda")


def _inputs(vam, context=8, batch=2, seed=0, steps=1):
    """Visual tokens for ``context`` frames and a command per predicted action
    block (inference predicts one block; training predicts ``context``)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens = torch.randint(0, 16384, (batch, context, 18, 32), generator=g, device="cuda")
    command = torch.full((batch, steps), 2, device="cuda", dtype=torch.long)
    return tokens, command


def _sample(vam, tokens, command, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return vam(tokens, command, torch.float16)


def test_zero_gate_reproduces_stock_vavam_exactly(vam):
    tokens, command = _inputs(vam)
    stock = _sample(vam, tokens, command)
    encoder = attach_conditioning(vam)
    adapter = GatedAdapter(state_dim=64).cuda()
    encoder.delta = adapter(torch.randn(2, 1, 64, device="cuda"))  # inference: one action block
    try:
        conditioned = _sample(vam, tokens, command)
    finally:
        encoder.delta = None
    assert torch.equal(stock, conditioned)


def test_open_gate_changes_the_trajectory(vam):
    tokens, command = _inputs(vam)
    encoder = attach_conditioning(vam)
    base = _sample(vam, tokens, command)
    adapter = GatedAdapter(state_dim=64).cuda()
    with torch.no_grad():
        adapter.gate.fill_(1.0)
    encoder.delta = adapter(torch.randn(2, 1, 64, device="cuda"))
    try:
        moved = _sample(vam, tokens, command)
    finally:
        encoder.delta = None
    assert not torch.allclose(base, moved)


def test_flow_loss_trains_only_the_adapter(vam):
    """Gradients reach the adapter (projection and gate) with VaVAM frozen."""
    vam.requires_grad_(False)
    encoder = attach_conditioning(vam)
    adapter = GatedAdapter(state_dim=64).cuda()
    tokens, command = _inputs(vam, context=8, batch=1, steps=8)
    actions = torch.randn(1, 8, vam.action_horizon, vam.action_dim, device="cuda") * 10
    t = torch.rand(1, 8, device="cuda")
    encoder.delta = adapter(torch.randn(1, 8, 64, device="cuda"))
    try:
        loss = (
            vam.forward_train_loss(tokens, command, actions, t)
            if hasattr(vam, "forward_train_loss")
            else type(vam).__mro__[1].forward(vam, tokens, command, actions, t)
        )
        loss.backward()
    finally:
        encoder.delta = None
        vam.eval()
    assert torch.isfinite(loss)
    assert adapter.gate.grad is not None and adapter.gate.grad.abs() > 0
    # At gate 0 the projection gets no gradient yet; it does once the gate opens.
    assert all(p.grad is None for p in vam.parameters())
