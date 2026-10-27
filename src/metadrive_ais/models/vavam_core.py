"""VaVAM-B frame preprocessing and VQ tokenization without AlpaSim's driver package.

``alpasim_driver.models`` imports every driver model (Alpamayo 1 / 1.5 / 2), so
environments without those dependencies (e.g. AlpaGym's) use this module. It
reproduces ``alpasim_driver.models.vam_model.VAMModel`` exactly: PIL bilinear
resize to height 900, center crop to 1600 wide, ``NeuroNCAPTransform``, then
the JIT-compiled LlamaGen VQ encoder under fp16 autocast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import omegaconf.dictconfig
import omegaconf.listconfig
import torch
import torch.serialization
from PIL import Image

# Let torch.load recreate the OmegaConf containers embedded in VaVAM checkpoints
# (as alpasim_driver.models.vam_model does).
torch.serialization.add_safe_globals(
    [omegaconf.listconfig.ListConfig, omegaconf.dictconfig.DictConfig]
)

ASSETS = Path.home() / "ext/alpasim/data/drivers/vavam"
TOKENIZER = ASSETS / "VQ_ds16_16384_llamagen_encoder.jit"
INPUT_HW = (900, 1600)  # NeuroNCAPTransform's expected input
COMMANDS = {"RIGHT": 0, "LEFT": 1, "STRAIGHT": 2}  # VaVAM command ids


def resize_and_center_crop(image: np.ndarray, height: int = 900, width: int = 1600) -> np.ndarray:
    """AlpaSim's ``BaseTrajectoryModel._resize_and_center_crop`` (HWC uint8)."""
    h, w = image.shape[:2]
    if (h, w) == (height, width):
        return image
    pil = Image.fromarray(image)
    new_w = int(w * height / h)
    pil = pil.resize((new_w, height), Image.Resampling.BILINEAR)
    if new_w > width:
        left = (new_w - width) // 2
        pil = pil.crop((left, 0, left + width, height))
    elif new_w < width:
        raise ValueError(f"image width {new_w} too small after resize, need {width}")
    return np.array(pil)


class VaVAMTokenizer:
    """Frames (HWC uint8) -> VaVAM VQ tokens [N, 18, 32]."""

    def __init__(self, device, tokenizer_path: str | Path = TOKENIZER):
        from vam.datalib.transforms import NeuroNCAPTransform

        self.device = torch.device(device)
        self.encoder = torch.jit.load(str(tokenizer_path), map_location=self.device).eval()
        self.transform = NeuroNCAPTransform()

    @torch.no_grad()
    def __call__(self, frames: list[np.ndarray]) -> torch.Tensor:
        batch = torch.stack([self.transform(resize_and_center_crop(f)) for f in frames])
        with torch.autocast(self.device.type, dtype=torch.float16):
            return self.encoder(batch.to(self.device))


def route_command(route_xy: np.ndarray, threshold_m: float = 3.0, lookahead_m: float = 20.0) -> int:
    """AlpaSim's route -> command rule on rig-frame waypoints (NaN rows ignored)."""
    for x, y in route_xy:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if np.hypot(x, y) >= lookahead_m:
            if y > threshold_m:
                return COMMANDS["LEFT"]
            if y < -threshold_m:
                return COMMANDS["RIGHT"]
            return COMMANDS["STRAIGHT"]
    return COMMANDS["STRAIGHT"]
