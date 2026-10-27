"""Load VaVAM-B as trained: with its muP base shapes.

AlpaSim's ``load_inference_VAM`` sets ``gpt_mup_base_shapes`` and
``action_mup_base_shapes`` to ``None``. The checkpoint was trained with
base shapes (``mup_shapes/*.bsh`` in valeoai/VideoActionModel); without them
muP's readout width multiplier defaults to 1, so the action expert's flow
velocity comes out 4x too large (action width 256 / base 64).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch

ASSETS = Path.home() / "ext/alpasim/data/drivers/vavam"
GPT_SHAPES = ASSETS / "mup_shapes/gpt2_24layers_basewidth256.bsh"
ACTION_SHAPES = (
    ASSETS / "mup_shapes/actionexpert_24layers_baseattentiondim256_baseembeddingdim64.bsh"
)


def load_vavam(
    checkpoint: str | Path = ASSETS / "VAM_width_1024_pretrained_139k.pt",
    device: str | torch.device = "cuda",
    mup_base_shapes: bool = True,
):
    """``VideoActionModelInference`` with (default) or without its muP base shapes."""
    from vam.action_expert import VideoActionModelInference

    import metadrive_ais.models.vavam_core  # noqa: F401  (registers OmegaConf safe globals)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)  # trusted release
    config = dict(ckpt["hyper_parameters"]["vam_conf"])
    config.pop("_target_", None)
    config.pop("_recursive_", None)
    config["gpt_checkpoint_path"] = None
    config["action_checkpoint_path"] = None
    config["gpt_mup_base_shapes"] = str(GPT_SHAPES) if mup_base_shapes else None
    config["action_mup_base_shapes"] = str(ACTION_SHAPES) if mup_base_shapes else None
    vam = VideoActionModelInference(**config)
    state = OrderedDict((k.replace("vam.", ""), v) for k, v in ckpt["state_dict"].items())
    vam.load_state_dict(state, strict=True)
    return vam.eval().to(device)
