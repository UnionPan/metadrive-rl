"""Side-by-side comparison video of several policies on the same scene.

Stacks AlpaSim's per-rollout evaluation videos (camera view, BEV with plans,
live metrics) horizontally, frame-synchronised, with a label per policy.

Usage:
    python -m metadrive_rl.visualization.compare out.mp4 LABEL=RUN_DIR [LABEL=RUN_DIR ...] \\
        --scene <scene-id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


def rollout_video(run_dir: Path, scene: str) -> Path:
    videos = sorted(run_dir.glob(f"**/rollouts/{scene}/*/*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no video for {scene} under {run_dir}")
    return videos[0]


def label(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 28), fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255))
    return np.asarray(img)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out", type=Path)
    parser.add_argument("runs", nargs="+", help="LABEL=RUN_DIR")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    streams = []
    for spec in args.runs:
        name, _, run = spec.partition("=")
        streams.append((name, iio.imread(rollout_video(Path(run), args.scene), index=None)))
    length = max(len(frames) for _, frames in streams)
    height = max(frames.shape[1] for _, frames in streams)
    out = []
    for i in range(length):
        row = []
        for name, frames in streams:
            # Rollouts end at their first incident; hold the last frame.
            f = frames[min(i, len(frames) - 1)]
            f = np.pad(f, ((0, height - f.shape[0]), (0, 0), (0, 0)))
            row.append(label(f, name))
        out.append(np.concatenate(row, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, np.stack(out), fps=args.fps, codec="libx264", quality=7)
    print(f"{args.out}: {length} frames, {out[0].shape[1]}x{out[0].shape[0]}")


if __name__ == "__main__":
    main()
