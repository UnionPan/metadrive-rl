"""Feasibility check for running an Alpamayo checkpoint on this hardware.

Loads the model with SDPA attention (FlashAttention 2 has no Turing kernels),
feeds one sample built from rendered camera frames and a straight ego history,
and reports load time, inference latency, peak GPU memory, the reasoning text,
and the predicted trajectory. Frames repeat one camera's images across the four
camera slots: enough for timing and memory, not for judging driving quality.

Usage: python scripts/bench_alpamayo.py --model alpamayo1 --frames-from <dense-images-dir>
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODELS = {
    "alpamayo1": "nvidia/Alpamayo-R1-10B",
    "alpamayo1_5": "nvidia/Alpamayo-1.5-10B",
}


def load(name: str, dtype: torch.dtype, device: str):
    if name == "alpamayo1":
        from alpamayo_r1 import helper
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1 as Model
    else:
        from alpamayo1_5 import helper
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5 as Model
    config = Model.config_class.from_pretrained(MODELS[name])
    config.attn_implementation = "sdpa"
    model = Model.from_pretrained(MODELS[name], config=config, dtype=dtype).to(device).eval()
    return model, helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="alpamayo1")
    parser.add_argument("--frames-from", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-generation-length", type=int, default=256)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    t = time.time()
    model, helper = load(args.model, dtype, "cuda")
    print(f"load {time.time() - t:.0f}s, weights {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    processor = helper.get_processor(model.tokenizer)

    paths = sorted(args.frames_from.glob("*.jpg"))[:4]
    frames = torch.stack(
        [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).permute(2, 0, 1) for p in paths]
    )  # (4 frames, 3, H, W)
    image_frames = frames[None].repeat(4, 1, 1, 1, 1)  # (4 cameras, 4 frames, 3, H, W)
    steps = np.arange(-15, 1, dtype=np.float32) * 0.8  # 8 m/s straight, 16 x 0.1 s
    history_xyz = torch.tensor(np.stack([steps, 0 * steps, 0 * steps], -1))[None, None]
    history_rot = torch.eye(3).expand(1, 1, 16, 3, 3).clone()

    if args.model == "alpamayo1":
        messages = helper.create_message(image_frames.flatten(0, 1))
    else:  # cross-left, front-wide, cross-right, front-tele in AlpaSim's index order
        messages = helper.create_message(
            image_frames.flatten(0, 1), torch.tensor([0, 1, 2, 6]), num_frames_per_camera=4
        )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    print("prompt tokens", inputs["input_ids"].shape[-1])
    torch.cuda.reset_peak_memory_stats()
    for k in range(args.runs):
        # The model pops entries from its input, so every run gets a fresh copy.
        data = helper.to_device(
            {
                "tokenized_data": dict(inputs),
                "ego_history_xyz": history_xyz,
                "ego_history_rot": history_rot,
            },
            "cuda",
        )
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=data,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=args.max_generation_length,
                return_extra=True,
            )
        torch.cuda.synchronize()
        end = xyz[0, 0, 0, -1].float().cpu().numpy()
        print(
            f"run {k}: {time.time() - t:.2f}s, finite {bool(torch.isfinite(xyz).all())}, "
            f"endpoint {np.round(end, 1).tolist()}"
        )
    print(f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print("reasoning:", extra["cot"][0])


if __name__ == "__main__":
    main()
