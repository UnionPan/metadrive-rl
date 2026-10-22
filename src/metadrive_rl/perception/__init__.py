"""Frozen visual encoders: camera frame -> fixed-size embedding."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

ENCODERS = {
    # name: (HF model id, input (width, height); multiples of the 14 px patch)
    "dinov2-small": ("facebook/dinov2-small", (448, 280)),
    "dinov2-base": ("facebook/dinov2-base", (448, 280)),
}


class FrozenEncoder:
    """DINOv2 backbone in FP16 (Turing has no BF16); returns [CLS, mean(patches)]."""

    def __init__(self, name: str, device: torch.device | str):
        from transformers import AutoModel

        model_id, self.input_wh = ENCODERS[name]
        self.name = name
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(model_id, dtype=torch.float16)
        self.model.eval().requires_grad_(False).to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @property
    def dim(self) -> int:
        return 2 * self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """``images``: HWC uint8 arrays. Returns (N, dim) float32."""
        batch = np.stack(
            [np.asarray(Image.fromarray(im).resize(self.input_wh, Image.BILINEAR)) for im in images]
        )
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - self.mean) / self.std).half()
        tokens = self.model(pixel_values=x).last_hidden_state
        features = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1)
        return features.float().cpu().numpy()


def build_encoder(cfg: dict[str, Any], device: torch.device | str) -> FrozenEncoder:
    return FrozenEncoder(cfg["name"], cfg.get("device", device))
