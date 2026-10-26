"""Cache frozen encoder features for every frame of an episode table.

Writes ``<steps>.<encoder>.npy`` (float16, one row per table row, zeros where a
row has no frame) and ``<steps>.<encoder>.json`` describing the encoder,
revision, preprocessing and the table it belongs to.

Usage:
    python -m metadrive_ais.vision.extract data/episodes/<name>/steps.parquet \\
        --encoder dinov3-small
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from metadrive_ais.vision.encoders import FrozenEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", type=Path)
    parser.add_argument("--encoder", default="dinov3-small")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    df = pl.read_parquet(args.steps)
    paths = df["image"].to_list()
    encoder = FrozenEncoder(args.encoder)
    out = np.zeros((len(paths), encoder.dim), dtype=np.float16)
    have = [i for i, p in enumerate(paths) if p]
    start = time.time()
    for k in range(0, len(have), args.batch_size):
        idx = have[k : k + args.batch_size]
        images = [np.asarray(Image.open(paths[i]).convert("RGB")) for i in idx]
        out[idx] = encoder.encode(images).astype(np.float16)
    stem = args.steps.with_suffix("")
    np.save(f"{stem}.{args.encoder}.npy", out)
    meta = {
        **encoder.describe(),
        "rows": len(paths),
        "rows_with_frame": len(have),
        "table": str(args.steps),
        "table_rows_sha1": hashlib.sha1(
            "\n".join(
                f"{e}|{s}" for e, s in zip(df["episode_id"], df["step"], strict=True)
            ).encode()
        ).hexdigest(),
        "frames_per_s": round(len(have) / max(time.time() - start, 1e-6), 1),
    }
    Path(f"{stem}.{args.encoder}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
