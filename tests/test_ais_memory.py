import math

import numpy as np
import polars as pl
import pytest
import torch

from metadrive_ais.memory.ais import AISHeads, MemoryEncoder, ais_losses
from metadrive_ais.memory.sequences import collate, episodes_from_table

KINDS = ("frame", "stack", "gru")


def _inputs(b=2, t=6, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, d, generator=g), torch.randn(b, t, 5, generator=g)


@pytest.mark.parametrize("kind", KINDS)
def test_memory_episode_reset(kind):
    """Changing one episode's inputs never changes another episode's states."""
    torch.manual_seed(0)
    enc = MemoryEncoder(16, kind=kind, state_dim=8, stack_k=3)
    e, u = _inputs()
    z = enc(e, u)
    e2 = e.clone()
    e2[0] += torch.randn_like(e2[0])  # a constant shift would vanish in LayerNorm
    z2 = enc(e2, u)
    assert torch.allclose(z[1], z2[1]) and not torch.allclose(z[0], z2[0])


@pytest.mark.parametrize("kind", KINDS)
def test_no_future_leakage(kind):
    """z_t depends only on inputs up to t."""
    torch.manual_seed(0)
    enc = MemoryEncoder(16, kind=kind, state_dim=8, stack_k=3)
    e, u = _inputs()
    z = enc(e, u)
    e2, u2 = e.clone(), u.clone()
    e2[:, 4:] += torch.randn_like(e2[:, 4:])
    u2[:, 4:] -= 2.0
    z2 = enc(e2, u2)
    assert torch.allclose(z[:, :4], z2[:, :4]) and not torch.allclose(z[:, 4:], z2[:, 4:])


def test_losses_are_masked_and_finite():
    torch.manual_seed(0)
    enc, heads = MemoryEncoder(16, state_dim=8), AISHeads(8, 16, hidden=16)
    e, u = _inputs()
    batch = {
        "embed": e,
        "motion": u,
        "action": torch.randn(2, 6, 12),
        "has_action": torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.bool),
        "collision_next": torch.tensor([[0, 0, 1, 0, 0, math.nan]] * 2, dtype=torch.float32),
        "offroad_next": torch.full((2, 6), math.nan),  # no valid targets at all
        "progress_gain": torch.rand(2, 6),
        "valid": torch.ones(2, 6, dtype=torch.bool),
    }
    losses = ais_losses(heads(enc(e, u), batch["action"]), batch, heads.embed_norm)
    assert all(torch.isfinite(v) for v in losses.values())
    assert losses["offroad"].item() == 0.0


def test_episodes_from_table_motion_and_targets():
    steps = pl.DataFrame(
        {
            "episode_id": ["a"] * 3,
            "scene_id": ["s"] * 3,
            "step": [0, 1, 2],
            "time_us": [0, 500_000, 1_000_000],
            "ego_x": [0.0, 0.0, 0.0],
            "ego_y": [0.0, 2.0, 4.0],  # driving along +y with heading +90 deg
            "ego_yaw": [math.pi / 2] * 3,
            "issued_xy": [[[0.0, 0.0]] + [[float(k), 0.0] for k in range(1, 7)]] * 2 + [[]],
            "outcome_collision_any": [0.0, 0.0, 1.0],
            "outcome_offroad": [0.0, 0.0, 0.0],
            "outcome_progress": [0.1, 0.2, 0.4],
        }
    )
    (ep,) = episodes_from_table(steps, np.eye(3, 4, dtype=np.float32))
    assert ep.motion[1, :2] == pytest.approx([2.0, 0.0], abs=1e-6)  # forward in the ego frame
    assert ep.motion[1, 4] == pytest.approx(4.0)  # 2 m in 0.5 s
    assert ep.has_action.tolist() == [True, True, False]
    assert ep.collision_next[:2].tolist() == [0.0, 1.0] and np.isnan(ep.collision_next[2])
    assert ep.progress_gain[:2] == pytest.approx([0.1, 0.2])
    batch = collate([ep, ep])
    assert batch["embed"].shape == (2, 3, 4) and batch["has_action"].sum() == 4
