"""Learned latent action channel into VaVAM-B's action expert (M5a).

The adapter residual gets a second term, ``W_c c_t``, driven by a latent
action ``c_t``:

- ``LatentPrior`` q_phi(c | z_t): diagonal Gaussian from the AIS state. This is
  the stochastic policy RL trains; ``log_prob`` is exact.
- ``LatentPosterior`` p_psi(c | z_t, future): sees the expert's next waypoints
  and is used only while learning the channel (CVAE-style, with the flow loss
  as reconstruction and KL(posterior || prior)).
- ``LatentDecoder``: ``W_c c`` in the action-token space.
- ``PCAPosterior``: an alternative, fixed encoder. The standardized top
  principal components of the expert's next waypoints, fitted on the training
  futures (``LatentChannel.fit_pca``). The decoder sees an informative latent
  from the first step, which avoids the posterior collapse a learned encoder
  hit in flow SFT.

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


class PCAPosterior(nn.Module):
    """Deterministic hindsight latent: standardized principal components of the future."""

    def __init__(self, latent_dim: int, horizon: int = 6):
        super().__init__()
        self.register_buffer("mean", torch.zeros(2 * horizon))
        self.register_buffer("components", torch.eye(latent_dim, 2 * horizon))
        self.register_buffer("scale", torch.ones(latent_dim))

    def fit(self, futures: torch.Tensor) -> torch.Tensor:
        """Fit on futures [N, 6, 2]; returns the explained-variance fraction per component."""
        x = futures.flatten(1).double() / FUTURE_SCALE
        mean = x.mean(0)
        _, sv, vt = torch.linalg.svd(x - mean, full_matrices=False)
        d = self.components.shape[0]
        var = sv**2 / (len(x) - 1)
        self.mean.copy_(mean.float())
        self.components.copy_(vt[:d].float())
        self.scale.copy_(var[:d].sqrt().float())
        return (var[:d] / var.sum()).float()

    def forward(self, z: torch.Tensor, future: torch.Tensor) -> Normal:
        x = future.flatten(-2) / FUTURE_SCALE - self.mean
        mean = (x @ self.components.T) / self.scale
        return Normal(mean, torch.full_like(mean, float(torch.tensor(LOG_STD_RANGE[0]).exp())))


class LatentDecoder(nn.Module):
    def __init__(self, latent_dim: int, token_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(latent_dim, token_dim, bias=False)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.proj(c)


class LatentChannel(nn.Module):
    def __init__(
        self, state_dim: int, latent_dim: int = 4, token_dim: int = 256, encoder: str = "mlp"
    ):
        super().__init__()
        if encoder not in ("mlp", "pca"):
            raise ValueError(encoder)
        self.latent_dim = latent_dim
        self.prior = LatentPrior(state_dim, latent_dim)
        self.posterior = (
            PCAPosterior(latent_dim) if encoder == "pca" else LatentPosterior(state_dim, latent_dim)
        )
        self.decoder = LatentDecoder(latent_dim, token_dim)
        self.config = {"state_dim": state_dim, "latent_dim": latent_dim, "token_dim": token_dim}
        if encoder != "mlp":  # older bundles have no "encoder" key
            self.config["encoder"] = encoder

    def fit_pca(self, futures: torch.Tensor) -> torch.Tensor:
        return self.posterior.fit(futures)
