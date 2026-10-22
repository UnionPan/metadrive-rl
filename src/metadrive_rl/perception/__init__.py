"""Frozen visual encoders: camera frame -> fixed-size embedding."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ENCODERS = {
    # name: (HF model id, input (width, height) as multiples of the patch size, pooling)
    "dinov2-small": ("facebook/dinov2-small", (448, 280), "cls_mean"),
    "dinov2-base": ("facebook/dinov2-base", (448, 280), "cls_mean"),
    # Keeps a coarse spatial grid of patch features (where / how large things
    # appear), needed for dense quantities such as the distance to a lead car.
    "dinov2-base-grid": ("facebook/dinov2-base", (448, 280), "grid"),
    "dinov2-base-grid6x8": ("facebook/dinov2-base", (448, 280), "grid6x8"),
    "dinov3-base": ("facebook/dinov3-vitb16-pretrain-lvd1689m", (448, 288), "cls_mean"),
    "dinov3-base-grid": ("facebook/dinov3-vitb16-pretrain-lvd1689m", (448, 288), "grid"),
    "dinov3-base-grid6x8": ("facebook/dinov3-vitb16-pretrain-lvd1689m", (448, 288), "grid6x8"),
}
GRID_HW = (3, 4)


def grid_hw(name: str) -> tuple[int, int] | None:
    """Pooled patch grid of an encoder ("grid" = GRID_HW, "gridHxW"), else None."""
    pooling = ENCODERS[name][2]
    if not pooling.startswith("grid"):
        return None
    if pooling == "grid":
        return GRID_HW
    h, w = pooling.removeprefix("grid").split("x")
    return int(h), int(w)


class FrozenEncoder:
    """DINO backbone in FP16 (Turing has no BF16).

    ``cls_mean`` pooling returns [CLS, mean(patches)]; ``grid``/``gridHxW``
    additionally append patch features average-pooled to that grid
    (channel-major: C x H x W).
    """

    def __init__(self, name: str, device: torch.device | str):
        from transformers import AutoModel

        model_id, self.input_wh, self.pooling = ENCODERS[name]
        self.grid_hw = grid_hw(name)
        self.name = name
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(model_id, dtype=torch.float16)
        self.model.eval().requires_grad_(False).to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @property
    def dim(self) -> int:
        cells = self.grid_hw[0] * self.grid_hw[1] if self.grid_hw else 0
        return (2 + cells) * self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """``images``: HWC uint8 arrays. Returns (N, dim) float32."""
        batch = np.stack(
            [np.asarray(Image.fromarray(im).resize(self.input_wh, Image.BILINEAR)) for im in images]
        )
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - self.mean) / self.std).half()
        tokens = self.model(pixel_values=x).last_hidden_state
        patch = self.model.config.patch_size
        h, w = self.input_wh[1] // patch, self.input_wh[0] // patch
        patches = tokens[:, -h * w :]  # DINOv3 puts register tokens before the patches
        parts = [tokens[:, 0], patches.mean(1)]
        if self.grid_hw:
            grid = patches.transpose(1, 2).reshape(len(x), -1, h, w)
            parts.append(F.adaptive_avg_pool2d(grid.float(), self.grid_hw).flatten(1).half())
        return torch.cat(parts, dim=-1).float().cpu().numpy()


def build_encoder(cfg: dict[str, Any], device: torch.device | str) -> FrozenEncoder:
    return FrozenEncoder(cfg["name"], cfg.get("device", device))
