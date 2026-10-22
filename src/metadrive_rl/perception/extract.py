"""Encode a decision dataset's images with a frozen encoder.

Usage: python -m metadrive_rl.perception.extract <dataset.parquet> --encoder dinov2-small

Writes ``<dataset>.<encoder>.npy`` with one row per dataset row, in order, and
reports throughput.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image

from . import FrozenEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--encoder", default="dinov2-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    paths = pl.read_parquet(args.dataset, columns=["image"])["image"].to_list()
    encoder = FrozenEncoder(args.encoder, args.device)
    out = np.empty((len(paths), encoder.dim), dtype=np.float32)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for i in range(0, len(paths), args.batch_size):
        images = [np.asarray(Image.open(p).convert("RGB")) for p in paths[i : i + args.batch_size]]
        out[i : i + len(images)] = encoder.encode(images)
    elapsed = time.time() - start
    target = args.dataset.with_suffix(f".{args.encoder}.npy")
    np.save(target, out)
    print(
        f"{len(paths)} images -> {target} ({out.shape[1]}-d), {len(paths) / elapsed:.1f} img/s, "
        f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB"
    )


if __name__ == "__main__":
    main()
