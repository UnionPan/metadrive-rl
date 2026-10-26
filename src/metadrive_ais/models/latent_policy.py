"""Learned latent action channel into VaVAM-B's action expert (M5a).

The adapter residual gets a second term, ``W_c c_t``, driven by a latent
action ``c_t``:

- ``LatentPrior`` q_phi(c | z_t): diagonal Gaussian from the AIS state. This is
  the stochastic policy RL trains; ``log_prob`` is exact.
- ``LatentPosterior`` p_psi(c | z_t, future): sees the expert's next waypoints
  and is used only while learning the channel (CVAE-style, with the flow loss
  as reconstruction and KL(posterior || prior)).
- ``LatentDecoder``: ``W_c c`` in the action-token space.

The score-function gradient of RL applies to the sampled latent ``c_t``; VaVAM's
flow sampler stays frozen and its noise is an environment input.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

FUTURE_SCALE = 30.0  # metres; same scale as the AIS action input
LOG_STD_RANGE = (-4.0, 1.0)


def _gaussian(params: torch.Tensor) -> Normal:
    mean, log_std = params.chunk(2, -1)
    return Normal(mean, log_std.clamp(*LOG_STD_RANGE).exp())


class LatentPrior(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * latent_dim)
        )

    def forward(self, z: torch.Tensor) -> Normal:
        return _gaussian(self.net(z))


class LatentPosterior(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, horizon: int = 6, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 2 * horizon, hidden), nn.GELU(), nn.Linear(hidden, 2 * latent_dim)
        )

    def forward(self, z: torch.Tensor, future: torch.Tensor) -> Normal:
        return _gaussian(self.net(torch.cat([z, future.flatten(-2) / FUTURE_SCALE], -1)))


class LatentDecoder(nn.Module):
    def __init__(self, latent_dim: int, token_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(latent_dim, token_dim, bias=False)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.proj(c)


class LatentChannel(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int = 4, token_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.prior = LatentPrior(state_dim, latent_dim)
        self.posterior = LatentPosterior(state_dim, latent_dim)
        self.decoder = LatentDecoder(latent_dim, token_dim)
        self.config = {"state_dim": state_dim, "latent_dim": latent_dim, "token_dim": token_dim}
