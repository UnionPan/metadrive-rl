"""Frozen visual encoders (DINOv3) for the AIS memory: frame -> embedding e_t.

Features are ``[CLS, mean(patches)]`` and, with ``grid``, the patch tokens
average-pooled to a coarse grid appended channel-major. Runs in fp16 (Turing
has no native bf16).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ENCODERS = {
    # name: (HF model id, input (width, height) in multiples of the 16-px patch, grid hw or None)
    "dinov3-small": ("facebook/dinov3-vits16-pretrain-lvd1689m", (448, 256), None),
    "dinov3-small-grid": ("facebook/dinov3-vits16-pretrain-lvd1689m", (448, 256), (3, 4)),
    "dinov3-base": ("facebook/dinov3-vitb16-pretrain-lvd1689m", (448, 256), None),
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class FrozenEncoder:
    def __init__(self, name: str, device: torch.device | str = "cuda"):
        from transformers import AutoModel

        self.name = name
        self.model_id, self.input_wh, self.grid_hw = ENCODERS[name]
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(self.model_id, dtype=torch.float16)
        self.model.eval().requires_grad_(False).to(self.device)
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)

    @property
    def dim(self) -> int:
        cells = self.grid_hw[0] * self.grid_hw[1] if self.grid_hw else 0
        return (2 + cells) * self.model.config.hidden_size

    def describe(self) -> dict:
        return {
            "encoder": self.name,
            "model_id": self.model_id,
            "revision": self.revision,
            "input_wh": list(self.input_wh),
            "resize": "PIL bilinear, no crop",
            "normalize": {"mean": MEAN, "std": STD},
            "features": "[CLS, mean(patches)]"
            + (f" + grid {self.grid_hw}" if self.grid_hw else ""),
            "dtype": "float16",
            "dim": self.dim,
        }

    @torch.no_grad()
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """HWC uint8 frames -> (N, dim) float32."""
        batch = np.stack(
            [np.asarray(Image.fromarray(im).resize(self.input_wh, Image.BILINEAR)) for im in images]
        )
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - self.mean) / self.std).half()
        tokens = self.model(pixel_values=x).last_hidden_state
        h, w = self.input_wh[1] // 16, self.input_wh[0] // 16
        patches = tokens[:, -h * w :]  # registers sit between CLS and the patches
        parts = [tokens[:, 0], patches.mean(1)]
        if self.grid_hw:
            grid = patches.transpose(1, 2).reshape(len(x), -1, h, w)
            parts.append(F.adaptive_avg_pool2d(grid.float(), self.grid_hw).flatten(1).half())
        return torch.cat(parts, -1).float().cpu().numpy()
