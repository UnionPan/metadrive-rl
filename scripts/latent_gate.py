"""M5a gate metrics of a VaVAM-AIS bundle's latent channel on the held-out windows.

Latent spread at the 3 s waypoint over prior samples with VaVAM's flow noise
fixed, plus held-out flow loss and open-loop error of the prior-sampled policy
(``metadrive_ais.training.flow_sft.evaluate``), for the same 12 held-out train
scenes as flow SFT.

Usage: python scripts/latent_gate.py <bundle.pt> [--windows 128]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metadrive_ais.memory.ais import MemoryEncoder
from metadrive_ais.models.latent_policy import LatentChannel
from metadrive_ais.models.vavam_conditioning import GatedAdapter
from metadrive_ais.models.vavam_loading import load_vavam
from metadrive_ais.training import flow_sft as fs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("bundle", type=Path)
parser.add_argument(
    "--data", type=Path, default=Path("data/episodes/train_expert_replay/steps.parquet")
)
parser.add_argument("--windows", type=int, default=128)
args = parser.parse_args()

device = torch.device("cuda")
bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
episodes = fs.load_expert([args.data], bundle["feature_encoder"])
scenes = sorted({e.seq.scene_id for e in episodes})
held = set(np.random.default_rng(0).permutation(scenes)[:12].tolist())
val = [e for e in episodes if e.seq.scene_id in held]
wins = fs.windows(val)
wins = wins[:: max(1, len(wins) // args.windows)][: args.windows]

memory = MemoryEncoder(**bundle["memory"]["config"])
memory.load_state_dict(bundle["memory"]["state_dict"])
adapter = GatedAdapter(bundle["adapter"]["state_dim"], bundle["adapter"]["token_dim"])
adapter.load_state_dict(bundle["adapter"]["state_dict"])
latent = LatentChannel(**bundle["latent"]["config"])
latent.load_state_dict(bundle["latent"]["state_dict"])
trainer = fs.Trainer(
    load_vavam(device=device).requires_grad_(False),
    memory.to(device).eval(),
    adapter.to(device).eval(),
    None,
    device,
    latent.to(device).eval(),
)
report = fs.evaluate(trainer, val, wins)
keys = (
    "windows",
    "flow_loss",
    "flow_loss_zero_gate",
    "latent_spread_final_m",
    "latent_spread_lateral_m",
    "latent_spread_longitudinal_m",
)
print(
    json.dumps(
        {k: report[k] for k in keys}
        | {
            "ade_m": report["open_loop"]["ade_m"],
            "ade_m_zero_gate": report["open_loop_zero_gate"]["ade_m"],
        },
        indent=1,
    )
)
