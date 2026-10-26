"""Summarise flow-SFT runs (runs/sft/*/summary.json) for the memory ablation.

Each run contributes its best held-out epoch (lowest flow loss); epoch 0 is the
zero-gate stock model, identical across runs.

Usage: python scripts/sft_ablation_report.py runs/sft/abl_frame_s0 runs/sft/abl_gru_s0 ...
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def agg(values) -> str:
    v = np.array(list(values), dtype=float)
    return f"{v.mean():.3f}" + (f" ± {v.std():.3f}" if len(v) > 1 else "")


rows = defaultdict(list)
stock = None
for run in map(Path, sys.argv[1:]):
    f = run / "summary.json"
    if not f.exists():
        continue
    s = json.loads(f.read_text())
    stock = s["history"][0]
    rows[s["args"]["memory"]].append(min(s["history"][1:], key=lambda h: h["flow_loss"]))

z = stock["open_loop_zero_gate"]
print(
    f"Stock VaVAM-B (zero gate): held-out flow {stock['flow_loss_zero_gate']:.4f}, "
    f"ADE {z['ade_m']:.2f} m, FDE {z['fde_m']:.2f} m, "
    f"longitudinal median {z.get('longitudinal_final_median_m', float('nan')):+.2f} m\n"
)
columns = ["held-out flow", "ADE m", "FDE m", "lateral med. m", "longitudinal med. m", "best epoch"]
print("| memory | runs | " + " | ".join(columns) + " |")
print("|---|---|---|---|---|---|---|---|")
for memory, best in rows.items():
    ol = [b["open_loop"] for b in best]
    cells = [
        agg(b["flow_loss"] for b in best),
        agg(o["ade_m"] for o in ol),
        agg(o["fde_m"] for o in ol),
        agg(o["lateral_final_median_m"] for o in ol),
        agg(o.get("longitudinal_final_median_m", float("nan")) for o in ol),
        agg(b["epoch"] for b in best),
    ]
    print(f"| {memory} | {len(best)} | " + " | ".join(cells) + " |")
