"""Is VaVAM-B's lateral bias in the images or in the model?

Runs VaVAM-B (1 frame, AlpaSim's preprocessing) on the camera images an
AlpaSim run logged during the forced log replay, as logged and mirrored
left-right, with the STRAIGHT command and identical sampling noise. Reports
the final-waypoint lateral offset (y, positive = left). An image-driven bias
mirrors (y_flipped ~ -y); a model/prior bias keeps its sign.

Usage: python scripts/vavam_mirror_probe.py <alpasim-run-dir> [--frames 8]
"""

from __future__ import annotations

import argparse
import io

import numpy as np
import torch
from alpasim_driver.models.vam_model import VAMModel
from PIL import Image

from metadrive_ais.data.asl import find_rollouts, iter_entries

ASSETS = "/home/union/ext/alpasim/data/drivers/vavam"


def logged_images(path, frames: int) -> list[np.ndarray]:
    out = []
    for e in iter_entries(path):
        if e.WhichOneof("log_entry") == "driver_camera_image":
            data = e.driver_camera_image.camera_image.image_bytes
            out.append(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")))
            if len(out) == frames:
                break
    return out


def final_lateral(model: VAMModel, images: list[np.ndarray], seed: int = 0) -> np.ndarray:
    tokens = model._tokenize_frames(images)
    tokens = tokens.reshape(len(images), 1, *tokens.shape[1:])
    command = torch.full((len(images), 1), 2, device=model._device, dtype=torch.long)
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=model.DTYPE):
        traj = model._vam(tokens, command, model.DTYPE)  # (B, 1, 6, 2), x forward, y left
    return traj.float().reshape(len(images), -1, 2)[:, -1, 1].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    model = VAMModel(
        checkpoint_path=f"{ASSETS}/VAM_width_1024_pretrained_139k.pt",
        tokenizer_path=f"{ASSETS}/VQ_ds16_16384_llamagen_encoder.jit",
        device=torch.device("cuda"),
        camera_ids=["camera_front_wide_120fov"],
        context_length=1,
    )
    y, y_flip = [], []
    for path in find_rollouts(args.run_dir):
        images = logged_images(path, args.frames)
        if not images:
            continue
        y.append(final_lateral(model, images))
        y_flip.append(final_lateral(model, [np.ascontiguousarray(im[:, ::-1]) for im in images]))
    y, y_flip = np.concatenate(y), np.concatenate(y_flip)
    print(f"{len(y)} logged frames (forced replay, ego on the human path)")
    for name, v in (("as logged", y), ("mirrored", y_flip)):
        print(
            f"  {name:9s}: final lateral mean {v.mean():+.2f} m, "
            f"median {np.median(v):+.2f} m, right {np.mean(v < 0):.0%}"
        )
    print(f"  correlation(y, -y_mirrored) = {np.corrcoef(y, -y_flip)[0, 1]:+.2f}")


if __name__ == "__main__":
    main()
