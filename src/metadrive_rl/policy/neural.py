"""Trainable high-level policy: frozen image features + state -> action logits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from metadrive_rl.planner import NUM_ACTIONS

from .base import Observation
from .features import STATE_DIM, state_features


class PolicyHead(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim + STATE_DIM),
            nn.Linear(feature_dim + STATE_DIM, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, NUM_ACTIONS),
        )
        self.config = {"feature_dim": feature_dim, "hidden": hidden, "dropout": dropout}

    def forward(self, image_features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([image_features, state], dim=-1))


def save_checkpoint(path: Path, head: PolicyHead, encoder: str, version: str, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "head_config": head.config, "encoder": encoder,
                "version": version, **extra}, path)


class NeuralPolicy:
    """Policy protocol implementation; the plugin supplies ``obs.features``."""

    def __init__(self, head: PolicyHead, encoder_name: str, version: str,
                 temperature: float = 1.0, device: str = "cpu", context: int = 1):
        self.head = head.eval().to(device)
        self.encoder_name = encoder_name
        # Number of decisions' image features per input (frame-stack memory);
        # the plugin supplies them concatenated, newest first.
        self.context = context
        self.version = version
        self.temperature = temperature
        self.device = device

    @torch.no_grad()
    def distribution(self, obs: Observation) -> np.ndarray:
        if obs.features is None:
            raise ValueError(f"{self.version} needs image features; configure the encoder")
        feats = torch.as_tensor(obs.features, dtype=torch.float32, device=self.device)[None]
        state = torch.as_tensor(
            state_features(obs.speed_mps, obs.command, obs.lane_offset_m, obs.route_xy),
            device=self.device,
        )[None]
        logits = self.head(feats, state)[0] / self.temperature
        return torch.softmax(logits.double(), -1).cpu().numpy()


def load_neural_policy(checkpoint: str, temperature: float = 1.0) -> NeuralPolicy:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    head = PolicyHead(**ckpt["head_config"])
    head.load_state_dict(ckpt["state_dict"])
    return NeuralPolicy(head, ckpt["encoder"], ckpt["version"], temperature,
                        context=ckpt.get("context", 1))
