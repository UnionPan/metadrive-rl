"""Memory encoders and approximate-information-state (AIS) heads.

All variants share the observation projection and the heads, so capacity is
matched and only the history mechanism differs:

- ``frame``: z_t from (e_t, u_{t-1}) only (no memory)
- ``stack``: z_t from the last ``k`` (e, u) pairs (finite memory)
- ``gru``:   z_t = GRU(z_{t-1}, (e_t, u_{t-1})), reset at every episode start

AIS heads take (z_t, a_t) and predict next-step reward components (collision,
offroad: Bernoulli; progress gain: regression) and the next frozen embedding
e_{t+1} as a diagonal Gaussian. The embedding target is the frozen DINOv3
feature, so the predictive objective cannot collapse to a trivial constant.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MOTION_DIM = 5
ACTION_DIM = 12
MOTION_SCALE = torch.tensor([5.0, 1.0, 0.2, 0.5, 15.0])  # rough unit scales
ACTION_SCALE = 30.0


class MemoryEncoder(nn.Module):
    def __init__(self, embed_dim: int, kind: str = "gru", state_dim: int = 256, stack_k: int = 4):
        super().__init__()
        if kind not in ("frame", "stack", "gru"):
            raise ValueError(kind)
        self.kind, self.stack_k, self.state_dim = kind, stack_k, state_dim
        self.obs = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, state_dim), nn.GELU()
        )
        step_dim = state_dim + MOTION_DIM
        if kind == "gru":
            self.cell = nn.GRUCell(step_dim, state_dim)
        else:
            width = step_dim * (stack_k if kind == "stack" else 1)
            self.mix = nn.Sequential(
                nn.Linear(width, state_dim), nn.GELU(), nn.Linear(state_dim, state_dim)
            )
        self.config = {
            "embed_dim": embed_dim,
            "kind": kind,
            "state_dim": state_dim,
            "stack_k": stack_k,
        }

    def forward(self, embed: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        """[B, T, D], [B, T, 5] -> z [B, T, state_dim]; each sequence starts fresh."""
        x = torch.cat([self.obs(embed), motion / MOTION_SCALE.to(motion)], -1)
        if self.kind == "frame":
            return self.mix(x)
        if self.kind == "stack":
            k = self.stack_k
            padded = F.pad(x, (0, 0, k - 1, 0))  # zeros before the episode start
            window = torch.cat([padded[:, i : i + x.shape[1]] for i in range(k)], -1)
            return self.mix(window)
        z = x.new_zeros(x.shape[0], self.state_dim)
        states = []
        for t in range(x.shape[1]):
            z = self.cell(x[:, t], z)
            states.append(z)
        return torch.stack(states, 1)

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        """Empty history for ``batch`` new episodes: [B, k, step] (stack) or [B, state] (gru)."""
        step_dim = self.state_dim + MOTION_DIM
        if self.kind == "stack":
            return torch.zeros(batch, self.stack_k, step_dim, device=device)
        return torch.zeros(batch, self.state_dim, device=device)

    def step(
        self, state: torch.Tensor, embed: torch.Tensor, motion: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One query: (state, e_t [B, D], u_{t-1} [B, 5]) -> (z_t, new state).

        Matches ``forward`` on the whole sequence step by step.
        """
        x = torch.cat([self.obs(embed), motion / MOTION_SCALE.to(motion)], -1)
        if self.kind == "frame":
            return self.mix(x), state
        if self.kind == "stack":
            window = torch.cat([state[:, 1:], x[:, None]], 1)
            return self.mix(window.flatten(1)), window
        z = self.cell(x, state)
        return z, z


class AISHeads(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int, hidden: int = 256):
        super().__init__()

        def mlp(out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(state_dim + ACTION_DIM, hidden), nn.GELU(), nn.Linear(hidden, out)
            )

        self.reward = mlp(3)  # collision logit, offroad logit, progress gain
        self.next_embed = mlp(2 * embed_dim)  # mean, log-variance
        self.embed_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.cat([z, action / ACTION_SCALE], -1)
        reward = self.reward(h)
        mean, log_var = self.next_embed(h).chunk(2, -1)
        return {
            "collision_logit": reward[..., 0],
            "offroad_logit": reward[..., 1],
            "progress_gain": reward[..., 2],
            "next_mean": mean,
            "next_log_var": log_var.clamp(-6, 6),
        }


def ais_losses(out: dict, batch: dict, embed_norm: nn.LayerNorm) -> dict[str, torch.Tensor]:
    """Masked losses; a step counts when it has an action and finite targets."""
    m = batch["has_action"]
    losses = {}
    for name, key in (("collision", "collision_next"), ("offroad", "offroad_next")):
        target = batch[key]
        ok = m & torch.isfinite(target)
        losses[name] = (
            F.binary_cross_entropy_with_logits(out[f"{name}_logit"][ok], target[ok].clamp(0, 1))
            if ok.any()
            else out[f"{name}_logit"].sum() * 0
        )
    gain = batch["progress_gain"]
    ok = m & torch.isfinite(gain)
    losses["progress"] = (
        F.smooth_l1_loss(out["progress_gain"][ok] * 10, gain[ok] * 10)
        if ok.any()
        else gain.new_zeros(())
    )
    # Next embedding: e_{t+1} is the next row; the last step has no target.
    target = embed_norm(batch["embed"][:, 1:])
    ok = m[:, :-1] & batch["valid"][:, 1:]
    mean, log_var = out["next_mean"][:, :-1], out["next_log_var"][:, :-1]
    nll = 0.5 * (log_var + (target - mean) ** 2 / log_var.exp()).mean(-1)
    losses["next_embed"] = nll[ok].mean() if ok.any() else nll.sum() * 0
    return losses
