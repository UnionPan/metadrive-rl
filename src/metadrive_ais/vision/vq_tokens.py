"""Cache VaVAM's VQ tokens for every frame of an episode table.

Uses AlpaSim's ``VAMModel`` preprocessing and tokenizer, so tokens are exactly
what the driver computes in closed loop. Writes ``<steps>.vq.npy`` (int16,
``[rows, 18, 32]``; rows without a frame are -1) and ``<steps>.vq.json``.

Usage:
    python -m metadrive_ais.vision.vq_tokens data/episodes/<name>/steps.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image

ASSETS = Path.home() / "ext/alpasim/data/drivers/vavam"


def load_vam_model(device="cuda", context_length: int = 8):
    from alpasim_driver.models.vam_model import VAMModel

    return VAMModel(
        checkpoint_path=str(ASSETS / "VAM_width_1024_pretrained_139k.pt"),
        tokenizer_path=str(ASSETS / "VQ_ds16_16384_llamagen_encoder.jit"),
        device=torch.device(device),
        camera_ids=["camera_front_wide_120fov"],
        context_length=context_length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    df = pl.read_parquet(args.steps)
    paths = df["image"].to_list()
    model = load_vam_model()
    tokens = None
    have = [i for i, p in enumerate(paths) if p]
    for k in range(0, len(have), args.batch_size):
        idx = have[k : k + args.batch_size]
        out = model._tokenize_frames([np.asarray(Image.open(paths[i]).convert("RGB")) for i in idx])
        out = out.cpu().numpy().astype(np.int16)
        if tokens is None:
            tokens = np.full((len(paths), *out.shape[1:]), -1, dtype=np.int16)
        tokens[idx] = out
    stem = args.steps.with_suffix("")
    np.save(f"{stem}.vq.npy", tokens)
    meta = {
        "tokenizer": "VQ_ds16_16384_llamagen_encoder.jit (VideoActionModel v1.0.0)",
        "preprocessing": "alpasim_driver VAMModel._tokenize_frames",
        "shape": list(tokens.shape),
        "table": str(args.steps),
        "table_rows_sha1": hashlib.sha1(
            "\n".join(
                f"{e}|{s}" for e, s in zip(df["episode_id"], df["step"], strict=True)
            ).encode()
        ).hexdigest(),
    }
    Path(f"{stem}.vq.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
