"""M4 memory-dependent tests T1 and T2 (docs/PROGRESS.md), open loop on the
held-out expert-replay windows of flow SFT.

T1: GRU adapters with the memory re-run over only the last k queries at each
step (observation timestamps unchanged): held-out flow loss and ADE per k.
T2: windows sliced by the ego's speed history over the last 2 s (stop-and-go,
decelerating, accelerating, steady); per-window flow loss (noise and flow time
seeded per window, identical across models) and ADE per adapter, plus the
GRU − frame difference with a scene-cluster bootstrap CI.

Usage: python scripts/memory_tests.py --out reports/m4_memory_scenarios_open_loop.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metadrive_ais.evaluation.summarize import bootstrap_ci
from metadrive_ais.memory.ais import MemoryEncoder
from metadrive_ais.models.vavam_conditioning import GatedAdapter
from metadrive_ais.models.vavam_loading import load_vavam
from metadrive_ais.training import flow_sft as fs

MODELS = {
    "motion": "runs/sft/abl_motion_s0/best.pt",
    "frame": "runs/sft/abl_frame_s0/best.pt",
    "stack": "runs/sft/abl_stack_s0/best.pt",
    "gru16": "runs/sft/abl_gru16_s0/best.pt",
    "b4_gru16": "runs/sft/b4_gru16_s0/best.pt",
}
HISTORY = (1, 2, 4, 8, 16, None)


def speed_slice(e: fs.ExpertEpisode, end: int) -> str:
    v = e.seq.motion[:, 4]  # speed during the step before each query
    now, past, low = v[end], v[end - 4], v[end - 4 : end + 1].min()
    if low < 1.0 and now > 3.0:
        return "stop-and-go"
    if past - now > 2.0:
        return "decelerating"
    if now - past > 2.0:
        return "accelerating"
    return "steady"


def truncated(memory: MemoryEncoder, k: int | None):
    """``Trainer.states`` replacement: z_t from the last k queries only."""

    def states(self, episodes):
        batch = fs.collate([e.seq for e in episodes], self.device)
        embed, motion = batch["embed"], batch["motion"]
        if k is None:
            return memory(embed, motion), batch
        steps = [
            memory(embed[:, max(0, t - k + 1) : t + 1], motion[:, max(0, t - k + 1) : t + 1])[:, -1]
            for t in range(embed.shape[1])
        ]
        return torch.stack(steps, 1), batch

    return states


def per_window(trainer: fs.Trainer, episodes, wins) -> tuple[np.ndarray, np.ndarray]:
    """Flow loss (seeded per window) and ADE of each window."""
    flow = []
    for j, w in enumerate(wins):
        g = torch.Generator(device=trainer.device).manual_seed(j)
        t = torch.rand(1, fs.CONTEXT, generator=g, device=trainer.device)
        torch.manual_seed(j)
        with torch.no_grad():
            flow.append(trainer.flow_loss(episodes, [w], t)[0].item())
    ade = []
    for k in range(0, len(wins), 8):
        w = wins[k : k + 8]
        pred = trainer.sample(episodes, w, seed=k)
        truth = np.stack([episodes[i].future[end] for i, end in w])
        ade.extend(np.linalg.norm(pred - truth, axis=-1).mean(1))
    return np.array(flow), np.array(ade)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/episodes/train_expert_replay/steps.parquet")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    episodes = fs.load_expert([args.data], "dinov3-small")
    scenes = sorted({e.seq.scene_id for e in episodes})
    held = set(np.random.default_rng(0).permutation(scenes)[:12].tolist())
    val = [e for e in episodes if e.seq.scene_id in held]
    wins = fs.windows(val)
    slices = np.array([speed_slice(val[i], end) for i, end in wins])
    win_scene = np.array([val[i].seq.scene_id for i, _ in wins])
    vam = load_vavam(device=device).requires_grad_(False)

    trainers = {}
    for name, path in MODELS.items():
        ckpt = torch.load(path, map_location=device, weights_only=True)
        memory = MemoryEncoder(**ckpt["memory"]["config"]).to(device).eval()
        memory.load_state_dict(ckpt["memory"]["state_dict"])
        adapter = GatedAdapter(memory.state_dim).to(device).eval()
        adapter.load_state_dict(ckpt["adapter"])
        trainers[name] = fs.Trainer(vam, memory, adapter, None, device)
    result = {"windows": len(wins), "slices": {s: int((slices == s).sum()) for s in set(slices)}}

    # T1: history length for the GRU adapters.
    result["t1"] = {}
    for name in ("gru16", "b4_gru16"):
        trainer = trainers[name]
        rows = {}
        for k in HISTORY:
            trainer.states = truncated(trainer.memory, k).__get__(trainer)
            report = fs.evaluate(trainer, val, wins)
            rows["all" if k is None else str(k)] = {
                "flow_loss": report["flow_loss"],
                "ade_m": report["open_loop"]["ade_m"],
            }
            print(name, k, rows["all" if k is None else str(k)], flush=True)
        trainer.states = fs.Trainer.states.__get__(trainer)
        result["t1"][name] = rows

    # T2: per-window metrics, sliced by speed history.
    per = {}
    for name, trainer in trainers.items():
        per[name] = per_window(trainer, val, wins)
        print(name, "per-window done", flush=True)
    stock_flow = []
    for j, w in enumerate(wins):
        g = torch.Generator(device=device).manual_seed(j)
        t = torch.rand(1, fs.CONTEXT, generator=g, device=device)
        torch.manual_seed(j)
        with torch.no_grad():
            stock_flow.append(trainers["frame"].flow_loss(val, [w], t, gate_zero=True)[0].item())
    stock_flow = np.array(stock_flow)
    result["t2"] = {}
    for s in ["stop-and-go", "decelerating", "accelerating", "steady", "all"]:
        m = np.ones(len(wins), bool) if s == "all" else slices == s
        row = {"windows": int(m.sum()), "stock_flow": float(stock_flow[m].mean())}
        for name, (flow, ade) in per.items():
            row[name] = {"flow": float(flow[m].mean()), "ade_m": float(ade[m].mean())}
        for metric, idx in (("flow", 0), ("ade_m", 1)):
            d = per["gru16"][idx][m] - per["frame"][idx][m]
            sc = win_scene[m]
            scene_d = np.array([d[sc == x].mean() for x in sorted(set(sc))])
            lo, hi = bootstrap_ci(scene_d)
            row[f"gru16_minus_frame_{metric}"] = [float(d.mean()), lo, hi]
        result["t2"][s] = row
        print(s, row["windows"], row["gru16_minus_frame_flow"], flush=True)
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
