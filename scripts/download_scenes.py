"""Download the scenes listed in a splits file into AlpaSim's scene cache.

Files land in ``<alpasim-dir>/data/nre-artifacts/all-usdzs/<uuid>.usdz``, the
layout the AlpaSim wizard uses, so both the wizard and the gsplat renderer find
them. Requires accepted access to the gated HF dataset.

Usage: python scripts/download_scenes.py <splits.yaml> <alpasim-dir> [split ...]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("splits", type=Path)
    parser.add_argument("alpasim_dir", type=Path)
    parser.add_argument("names", nargs="*", default=["train", "val", "test"])
    args = parser.parse_args()

    out_dir = args.alpasim_dir / "data/nre-artifacts/all-usdzs"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = yaml.safe_load(args.splits.read_text())
    for name in args.names:
        for scene in splits[name]:
            target = out_dir / f"{scene['uuid']}.usdz"
            if target.exists():
                print(f"[{name}] have {scene['scene_id']}")
                continue
            staging = out_dir / ".staging"
            path = hf_hub_download(
                REPO,
                scene["hf_path"],
                repo_type="dataset",
                revision=scene["hf_revision"],
                local_dir=staging,
            )
            Path(path).rename(target)
            shutil.rmtree(staging)
            print(f"[{name}] downloaded {scene['scene_id']} ({target.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
