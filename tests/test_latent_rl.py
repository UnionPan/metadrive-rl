import copy

import pytest
import torch

pytest.importorskip("alpasim_grpc")

from metadrive_ais.models.latent_policy import LatentChannel  # noqa: E402
from metadrive_ais.training.latent_rl import advantages, ppo_update  # noqa: E402


def test_group_baseline_and_degenerate_scenes():
    scores = {
        "a1": ("A", 1.0),
        "a2": ("A", 0.0),
        "b1": ("B", 0.3),
        "b2": ("B", 0.3),  # all equal: no signal
        "c1": ("C", 0.7),  # a single rollout: no baseline
    }
    adv, stats = advantages(scores)
    assert set(adv) == {"a1", "a2"}
    assert adv["a1"] > 0 > adv["a2"] and adv["a1"] == pytest.approx(-adv["a2"])
    assert stats["degenerate_scenes"] == 2 and stats["scenes"] == 3


def test_ppo_moves_the_prior_toward_rewarded_latents():
    """Bandit check: reward latents with c[0] > 0; the prior mean of c[0] must rise."""
    torch.manual_seed(0)
    latent = LatentChannel(state_dim=8, latent_dim=2, token_dim=4)
    reference = copy.deepcopy(latent).requires_grad_(False)
    z = torch.randn(2048, 8)
    with torch.no_grad():
        dist = latent.prior(z)
        c = dist.sample()
        logp = dist.log_prob(c).sum(-1)
    a = torch.where(c[:, 0] > 0, 1.0, -1.0)
    before = float(latent.prior(z).mean[:, 0].mean())
    history = ppo_update(latent, reference, z, c, logp, a, epochs=4, lr=1e-3, ref_weight=0.0)
    after = float(latent.prior(z).mean[:, 0].mean())
    assert after > before + 0.05
    assert history[-1]["surrogate"] > 0 and history[-1]["kl_ref"] > 0
