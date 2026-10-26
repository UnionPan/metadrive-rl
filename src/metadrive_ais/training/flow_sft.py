"""Supervised fine-tuning of the AIS adapter with VaVAM-B's flow-matching loss.

Expert demonstrations are expert-replay episodes (the ego forced along the
human path): each query has VaVAM's VQ tokens, the route-derived command, the
frozen DINOv3 embedding, the executed motion, and the human's next six
waypoints (0.5 s apart, rig frame at the query) as the target. Training uses
windows of 8 consecutive queries, VaVAM-B's native context. The memory runs
over the whole episode prefix, and each window step gets the adapter residual
from its own state ``z_t``. VaVAM-B stays frozen; the adapter (and optionally
the memory and AIS heads) train.

Held-out scenes are scored by the flow loss and by open-loop samples against
the human future (ADE, FDE, signed lateral error), next to the zero-gate
(stock) model.

Usage:
    python -m metadrive_ais.training.flow_sft --data data/episodes/<expert>/steps.parquet \\
        --memory gru --out runs/sft/<name> [--ais-weight 0.5] [--epochs 20]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from metadrive_ais.memory.ais import KINDS, AISHeads, MemoryEncoder, ais_losses
from metadrive_ais.memory.sequences import Episode, collate, episodes_from_table
from metadrive_ais.models.vavam_conditioning import GatedAdapter, attach_conditioning

CONTEXT, HORIZON = 8, 6
COMMANDS = {"RIGHT": 0, "LEFT": 1, "STRAIGHT": 2, "UNKNOWN": 2}  # VaVAM's command ids


@dataclass
class ExpertEpisode:
    seq: Episode
    tokens: np.ndarray  # (T, 18, 32) int16
    command: np.ndarray  # (T,) VaVAM command ids
    future: np.ndarray  # (T, 6, 2) human waypoints in the rig frame at t
    has_future: np.ndarray  # (T,) bool


def human_futures(g: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x, y, yaw = (g[c].to_numpy() for c in ("ego_x", "ego_y", "ego_yaw"))
    T = len(g)
    fut = np.zeros((T, HORIZON, 2), dtype=np.float32)
    ok = np.zeros(T, dtype=bool)
    for t in range(T - HORIZON):
        c, s = np.cos(yaw[t]), np.sin(yaw[t])
        dx, dy = x[t + 1 : t + 1 + HORIZON] - x[t], y[t + 1 : t + 1 + HORIZON] - y[t]
        fut[t] = np.stack([c * dx + s * dy, -s * dx + c * dy], -1)
        ok[t] = True
    return fut, ok


def load_expert(tables: list[Path], encoder: str) -> list[ExpertEpisode]:
    out = []
    for table in tables:
        steps = pl.read_parquet(table)
        feats = np.load(table.with_suffix("").as_posix() + f".{encoder}.npy")
        tokens = np.load(table.with_suffix("").as_posix() + ".vq.npy")
        seqs = {
            e.episode_id: e
            for e in episodes_from_table(steps, feats, truncate_after_incident=False)
        }
        steps = steps.with_row_index("_row")
        for (episode_id,), g in steps.sort("episode_id", "step").group_by(
            "episode_id", maintain_order=True
        ):
            fut, ok = human_futures(g)
            rows = g["_row"].to_numpy()
            out.append(
                ExpertEpisode(
                    seq=seqs[episode_id],
                    tokens=tokens[rows],
                    command=np.array([COMMANDS[c] for c in g["command"]], dtype=np.int64),
                    future=fut,
                    has_future=ok & (tokens[rows, 0, 0] >= 0),
                )
            )
    return out


def windows(episodes: list[ExpertEpisode]) -> list[tuple[int, int]]:
    """(episode index, last step) of every 8-step window whose steps all have targets."""
    out = []
    for i, e in enumerate(episodes):
        for end in range(CONTEXT - 1, len(e.command)):
            if e.has_future[end - CONTEXT + 1 : end + 1].all():
                out.append((i, end))
    return out


class Trainer:
    def __init__(
        self, vam, memory: MemoryEncoder, adapter: GatedAdapter, heads: AISHeads | None, device
    ):
        self.vam, self.memory, self.adapter, self.heads, self.device = (
            vam,
            memory,
            adapter,
            heads,
            device,
        )
        self.conditioner = attach_conditioning(vam)

    def states(self, episodes: list[ExpertEpisode]) -> torch.Tensor:
        batch = collate([e.seq for e in episodes], self.device)
        return self.memory(batch["embed"], batch["motion"]), batch

    def flow_loss(self, episodes, wins, t_values=None, gate_zero=False):
        """Mean flow-matching loss over windows (one VaVAM forward per batch)."""
        eps = sorted({i for i, _ in wins})
        z, seq_batch = self.states([episodes[i] for i in eps])
        pos = {i: k for k, i in enumerate(eps)}
        tok, cmd, act, delta = [], [], [], []
        for i, end in wins:
            sl = slice(end - CONTEXT + 1, end + 1)
            e = episodes[i]
            tok.append(torch.from_numpy(e.tokens[sl].astype(np.int64)))
            cmd.append(torch.from_numpy(e.command[sl]))
            act.append(torch.from_numpy(e.future[sl]))
            delta.append(self.adapter(z[pos[i], sl]))
        tok = torch.stack(tok).to(self.device)
        cmd = torch.stack(cmd).to(self.device)
        act = torch.stack(act).to(self.device)
        t = t_values if t_values is not None else torch.rand(len(wins), CONTEXT, device=self.device)
        self.conditioner.delta = None if gate_zero else torch.stack(delta)
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = type(self.vam).__mro__[1].forward(self.vam, tok, cmd, act, t)
        finally:
            self.conditioner.delta = None
        aux = {}
        if self.heads is not None:
            aux = ais_losses(self.heads(z, seq_batch["action"]), seq_batch, self.heads.embed_norm)
        return loss.float(), aux

    @torch.no_grad()
    def sample(self, episodes, wins, gate_zero=False, seed=0):
        """Open-loop trajectory at each window's last step: (N, 6, 2)."""
        eps = sorted({i for i, _ in wins})
        z, _ = self.states([episodes[i] for i in eps])
        pos = {i: k for k, i in enumerate(eps)}
        tok = torch.stack(
            [
                torch.from_numpy(episodes[i].tokens[end - CONTEXT + 1 : end + 1].astype(np.int64))
                for i, end in wins
            ]
        ).to(self.device)
        cmd = torch.tensor([[episodes[i].command[end]] for i, end in wins], device=self.device)
        delta = torch.stack([self.adapter(z[pos[i], end]) for i, end in wins])[:, None]
        self.conditioner.delta = None if gate_zero else delta
        try:
            torch.manual_seed(seed)
            with torch.autocast("cuda", dtype=torch.float16):
                traj = self.vam(tok, cmd, torch.float16)
        finally:
            self.conditioner.delta = None
        return traj.float().reshape(len(wins), HORIZON, 2).cpu().numpy()


def open_loop_report(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = pred - truth
    d = np.linalg.norm(err, axis=-1)
    return {
        "ade_m": float(d.mean()),
        "fde_m": float(d[:, -1].mean()),
        "ade_median_m": float(np.median(d.mean(1))),
        "lateral_final_mean_m": float(err[:, -1, 1].mean()),
        "lateral_final_median_m": float(np.median(err[:, -1, 1])),
        "right_of_human": float((err[:, -1, 1] < 0).mean()),
        # Positive = ahead of the human (too fast), negative = behind (too slow).
        "longitudinal_final_median_m": float(np.median(err[:, -1, 0])),
        "longitudinal_final_mean_m": float(err[:, -1, 0].mean()),
    }


def evaluate(trainer: Trainer, episodes, wins, batch: int = 8) -> dict:
    trainer.memory.eval(), trainer.adapter.eval()
    g = torch.Generator(device=trainer.device).manual_seed(0)
    flow, flow0, preds, preds0, truth = [], [], [], [], []
    for k in range(0, len(wins), batch):
        w = wins[k : k + batch]
        t = torch.rand(len(w), CONTEXT, generator=g, device=trainer.device)
        with torch.no_grad():
            torch.manual_seed(k)
            flow.append(trainer.flow_loss(episodes, w, t)[0].item() * len(w))
            torch.manual_seed(k)
            flow0.append(trainer.flow_loss(episodes, w, t, gate_zero=True)[0].item() * len(w))
        preds.append(trainer.sample(episodes, w, seed=k))
        preds0.append(trainer.sample(episodes, w, gate_zero=True, seed=k))
        truth.append(np.stack([episodes[i].future[end] for i, end in w]))
    preds, preds0, truth = map(np.concatenate, (preds, preds0, truth))
    return {
        "windows": len(wins),
        "flow_loss": sum(flow) / len(wins),
        "flow_loss_zero_gate": sum(flow0) / len(wins),
        "open_loop": open_loop_report(preds, truth),
        "open_loop_zero_gate": open_loop_report(preds0, truth),
        "gate": float(trainer.adapter.gate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--encoder", default="dinov3-small")
    parser.add_argument("--memory", choices=KINDS, default="gru")
    parser.add_argument(
        "--memory-checkpoint", type=Path, help="AIS memory to start from (memory.train)"
    )
    parser.add_argument("--freeze-memory", action="store_true")
    parser.add_argument(
        "--ais-weight", type=float, default=0.0, help="AIS auxiliary losses (B4 > 0, B3 = 0)"
    )
    parser.add_argument("--holdout-scenes", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from metadrive_ais.models.vavam_loading import load_vavam

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    episodes = load_expert(args.data, args.encoder)
    scenes = sorted({e.seq.scene_id for e in episodes})
    held = set(np.random.default_rng(0).permutation(scenes)[: args.holdout_scenes].tolist())
    train_eps = [e for e in episodes if e.seq.scene_id not in held]
    val_eps = [e for e in episodes if e.seq.scene_id in held]
    train_w, val_w = windows(train_eps), windows(val_eps)

    # VaVAM as trained (muP base shapes); AlpaSim's loader inflates the flow velocity 4x.
    vam = load_vavam(device=device).requires_grad_(False)
    dim = episodes[0].seq.embed.shape[1]
    memory = MemoryEncoder(dim, args.memory).to(device)
    if args.memory_checkpoint:
        memory.load_state_dict(
            torch.load(args.memory_checkpoint, map_location=device, weights_only=True)["encoder"]
        )
    memory.requires_grad_(not args.freeze_memory)
    adapter = GatedAdapter(memory.state_dim).to(device)
    heads = AISHeads(memory.state_dim, dim).to(device) if args.ais_weight > 0 else None
    trainer = Trainer(vam, memory, adapter, heads, device)
    params = [
        p
        for m in (memory, adapter, heads)
        if m is not None
        for p in m.parameters()
        if p.requires_grad
    ]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    # Gradients pass through VaVAM's fp16 action expert: scale to avoid underflow.
    scaler = torch.amp.GradScaler("cuda")

    args.out.mkdir(parents=True, exist_ok=True)
    history = [{"epoch": 0, **evaluate(trainer, val_eps, val_w)}]
    print(json.dumps(history[-1]))
    best = (history[-1]["flow_loss"], 0)
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        memory.train(not args.freeze_memory), adapter.train()
        order = rng.permutation(len(train_w))
        for k in range(0, len(order), args.batch):
            w = [train_w[j] for j in order[k : k + args.batch]]
            loss, aux = trainer.flow_loss(train_eps, w)
            total = loss + args.ais_weight * sum(aux.values()) if aux else loss
            opt.zero_grad()
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        history.append({"epoch": epoch, **evaluate(trainer, val_eps, val_w)})
        print(json.dumps(history[-1]))
        if history[-1]["flow_loss"] < best[0]:
            best = (history[-1]["flow_loss"], epoch)
            torch.save(
                {
                    "memory": {"config": memory.config, "state_dict": memory.state_dict()},
                    "adapter": adapter.state_dict(),
                    "feature_encoder": args.encoder,
                    "epoch": epoch,
                },
                args.out / "best.pt",
            )
    summary = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "episodes": {"train": len(train_eps), "val": len(val_eps)},
        "windows": {"train": len(train_w), "val": len(val_w)},
        "holdout_scenes": sorted(held),
        "best_epoch": best[1],
        "history": history,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
