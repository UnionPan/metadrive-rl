"""History encoders for the AIS study (PROJECT_GUIDELINES §4).

All variants map a rollout's sequence of per-decision observations
``o_t = [e_t, s_t]`` (frozen image embedding, deployable state) to a summary
``z_t`` and score the 12 actions with the same cost head:

* ``frame``: z_t = MLP(o_t)                          (memoryless)
* ``stack``: z_t = MLP(o_t, o_{t-1}, ..., o_{t-K+1}) (fixed window)
* ``gru``:   z_t = GRU(z_{t-1}, o_t)                 (generic memory)
* ``ais``:   ``gru`` plus a transition head that predicts the next
  observation embedding from (z_t, a_t); the cost head plays the role of the
  reward head (per-action cost = -reward).

``z_t`` depends only on past and present observations (causal).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from metadrive_rl.planner import NUM_ACTIONS


class HistoryPolicy(nn.Module):
    def __init__(
        self,
        kind: str,
        obs_dim: int,
        hidden: int = 128,
        stack: int = 3,
        proj_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kind, self.stack = kind, stack
        self.inp = nn.Sequential(
            nn.LayerNorm(obs_dim), nn.Linear(obs_dim, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        if kind == "stack":
            self.mix = nn.Sequential(nn.Linear(stack * hidden, hidden), nn.GELU())
        elif kind in ("gru", "ais"):
            self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        elif kind != "frame":
            raise ValueError(kind)
        self.cost = nn.Linear(hidden, NUM_ACTIONS)
        if kind == "ais":
            self.transition = nn.Sequential(
                nn.Linear(hidden + NUM_ACTIONS, hidden), nn.GELU(), nn.Linear(hidden, proj_dim)
            )

    def summarise(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """obs (B, T, D), mask (B, T) -> z (B, T, H)."""
        h = self.inp(obs) * mask[..., None]
        if self.kind == "frame":
            return h
        if self.kind == "stack":
            past = [F.pad(h, (0, 0, k, 0))[:, : h.shape[1]] for k in range(self.stack)]
            return self.mix(torch.cat(past, -1))
        z, _ = self.rnn(h)
        return z

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.summarise(obs, mask)
        return self.cost(z), z

    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transition(torch.cat([z, F.one_hot(action, NUM_ACTIONS).float()], -1))
