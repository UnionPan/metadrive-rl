"""Condition VaVAM-B's action expert on an external state (AIS memory).

VaVAM's action tokens are built by ``ActionEncoder(noisy_actions, command,
diffusion_step)`` with shape ``[B, t, horizon, D]``. ``ConditionedActionEncoder``
wraps it and adds a residual ``delta[B, t, D]``, broadcast over the horizon
tokens of each timestep (where the navigation command already enters).
``GatedAdapter`` maps a state ``z`` to that residual through a learned
projection scaled by a gate initialised to exactly zero, so a fresh adapter
reproduces stock VaVAM-B bit for bit.
"""

from __future__ import annotations

import torch
from torch import nn


class ConditionedActionEncoder(nn.Module):
    """``ActionEncoder`` plus an externally set per-timestep residual."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.delta: torch.Tensor | None = None  # [B, t, D], set before each forward

    def forward(self, actions, high_level_command, diffusion_step):
        tokens = self.encoder(actions, high_level_command, diffusion_step)  # [B, t, H, D]
        if self.delta is None:
            return tokens
        if self.delta.shape[:2] != tokens.shape[:2]:
            raise ValueError(f"delta {tuple(self.delta.shape)} vs tokens {tuple(tokens.shape)}")
        return tokens + self.delta[:, :, None, :].to(tokens.dtype)


class GatedAdapter(nn.Module):
    """``delta = gate * W z``, gate initialised to zero."""

    def __init__(self, state_dim: int, token_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(state_dim, token_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.gate * self.proj(z)


def attach_conditioning(vam: nn.Module) -> ConditionedActionEncoder:
    """Wrap the action encoder of a loaded VaVAM in place; return the wrapper."""
    expert = vam.joint_model.action_expert
    if isinstance(expert.action_encoder, ConditionedActionEncoder):
        return expert.action_encoder
    expert.action_encoder = ConditionedActionEncoder(expert.action_encoder)
    return expert.action_encoder
