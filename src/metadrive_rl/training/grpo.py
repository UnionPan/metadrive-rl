"""Group-relative policy optimisation (GRPO) on AlpaSim closed-loop outcomes.

One iteration = collect -> build an immutable batch -> update:

* ``build``: read sampled rollouts (several per scene) and AlpaSim's aggregated
  scores. Every decision of a rollout gets the rollout's group-relative
  advantage ``(R - mean_g R) / (std_g R + eps)``, where the group is all
  rollouts of the same scene. The behaviour log-probability is the one logged
  at sampling time, for the action actually sampled.
* ``update``: clipped importance-ratio objective against those behaviour
  log-probs plus a KL penalty to a frozen reference policy (the SFT head).

Rewards are AlpaSim's verifiable per-rollout score (0 on an at-fault collision
or offroad, else clipped relative progress), optionally shaped by
``--progress-weight`` so near-identical groups still carry signal.

Usage:
    python -m metadrive_rl.training.grpo build <rollout-dir> <batch-dir> --features <dir>
    python -m metadrive_rl.training.grpo update <batch-dir> \\
        --policy <ckpt> --ref <ckpt> --out <ckpt> --version <name>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from metadrive_rl.policy.features import state_features
from metadrive_rl.policy.neural import PolicyHead, save_checkpoint
from metadrive_rl.training.sft import stack_context


def rollout_rewards(rollout_dir: Path, progress_weight: float) -> dict[str, tuple[str, float]]:
    """rollout_id -> (scene, reward)."""
    out = {}
    for f in rollout_dir.glob("**/aggregate/results-summary.json"):
        for r in json.loads(f.read_text())["rollouts"]:
            reward = r["score"] + progress_weight * r["metrics"]["progress"]
            out[r["rollout_id"]] = (r["clipgt_id"], float(reward))
    return out


def build(rollout_dir: Path, batch_dir: Path, feature_dir: Path, progress_weight: float) -> dict:
    from metadrive_rl.data.asl import find_rollouts, read_rollout

    rewards = rollout_rewards(rollout_dir, progress_weight)
    by_scene: dict[str, list[float]] = defaultdict(list)
    for scene, reward in rewards.values():
        by_scene[scene].append(reward)
    stats = {s: (np.mean(v), np.std(v)) for s, v in by_scene.items()}

    rows, feats = [], []
    for path in find_rollouts(rollout_dir):
        log = read_rollout(path)
        if log.rollout_uuid not in rewards:
            continue  # failed or retried rollout without an aggregated score
        scene, reward = rewards[log.rollout_uuid]
        mean, std = stats[scene]
        advantage = (reward - mean) / (std + 1e-6)
        for d in log.decisions:
            if not d["is_decision"]:
                continue
            f = feature_dir / f"{d['decision_id']}.npy"
            if not f.exists():
                continue
            feats.append(np.load(f))
            rows.append(
                {
                    "decision_id": d["decision_id"],
                    "scene_id": scene,
                    "rollout_uuid": log.rollout_uuid,
                    "time_us": d["time_us"],
                    "policy_version": d["policy_version"],
                    "action": d["action_index"],
                    "behaviour_logp": d["log_prob"],
                    "reward": reward,
                    "advantage": advantage,
                    "state": state_features(
                        d["speed_mps"], d["command"], d["lane_offset_m"], d.get("route_xy")
                    ).tolist(),
                }
            )
    batch_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(batch_dir / "decisions.parquet")
    np.save(batch_dir / "features.npy", np.stack(feats).astype(np.float32))
    degenerate = sum(1 for s, (m, sd) in stats.items() if sd < 1e-6)
    manifest = {
        "rollout_dir": str(rollout_dir),
        "rollouts": len(rewards),
        "decisions": len(rows),
        "scenes": len(by_scene),
        "groups_with_zero_reward_variance": degenerate,
        "reward_mean": float(np.mean([r for _, r in rewards.values()])),
        "policy_versions": sorted({r["policy_version"] for r in rows}),
        "progress_weight": progress_weight,
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_head(path: Path, device: torch.device) -> tuple[PolicyHead, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    head = PolicyHead(**ckpt["head_config"]).to(device)
    head.load_state_dict(ckpt["state_dict"])
    return head, ckpt


def update(
    batch_dir: Path,
    policy_path: Path,
    ref_path: Path,
    out: Path,
    *,
    clip: float,
    kl_coef: float,
    epochs: int,
    lr: float,
    batch_size: int,
    version: str,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pl.read_parquet(batch_dir / "decisions.parquet")
    policy, ckpt = load_head(policy_path, device)
    # Batches store single-decision features; rebuild the policy's frame stack.
    feats = stack_context(df, np.load(batch_dir / "features.npy"), ckpt.get("context", 1))
    x = torch.tensor(feats, device=device)
    s = torch.tensor(np.array(df["state"].to_list()), dtype=torch.float32, device=device)
    a = torch.tensor(df["action"].to_numpy(), device=device)
    old_logp = torch.tensor(df["behaviour_logp"].to_numpy(), dtype=torch.float32, device=device)
    adv = torch.tensor(df["advantage"].to_numpy(), dtype=torch.float32, device=device)

    reference, _ = load_head(ref_path, device)
    reference.eval().requires_grad_(False)
    with torch.no_grad():
        ref_logp_all = F.log_softmax(reference(x, s), -1)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(len(a), device=device)
        for i in range(0, len(a), batch_size):
            idx = perm[i : i + batch_size]
            logp_all = F.log_softmax(policy(x[idx], s[idx]), -1)
            logp = logp_all.gather(1, a[idx, None])[:, 0]
            ratio = torch.exp(logp - old_logp[idx])
            surrogate = torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx])
            kl = (logp_all.exp() * (logp_all - ref_logp_all[idx])).sum(-1)
            loss = -(surrogate - kl_coef * kl).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logp_all = F.log_softmax(policy(x, s), -1)
            ratio = torch.exp(logp_all.gather(1, a[:, None])[:, 0] - old_logp)
            history.append(
                {
                    "epoch": epoch + 1,
                    "kl_to_ref": float((logp_all.exp() * (logp_all - ref_logp_all)).sum(-1).mean()),
                    "entropy": float(-(logp_all.exp() * logp_all).sum(-1).mean()),
                    "clip_fraction": float(((ratio - 1).abs() > clip).float().mean()),
                    "ratio_mean": float(ratio.mean()),
                }
            )
    save_checkpoint(
        out,
        policy,
        ckpt["encoder"],
        version,
        {
            "context": ckpt.get("context", 1),
            "grpo": {
                "batch": str(batch_dir),
                "parent": str(policy_path),
                "reference": str(ref_path),
                "history": history,
            },
        },
    )
    return {"decisions": len(a), "advantage_std": float(adv.std()), "final": history[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("rollout_dir", type=Path)
    b.add_argument("batch_dir", type=Path)
    b.add_argument("--features", type=Path, required=True)
    b.add_argument("--progress-weight", type=float, default=0.0)
    u = sub.add_parser("update")
    u.add_argument("batch_dir", type=Path)
    u.add_argument("--policy", type=Path, required=True)
    u.add_argument("--ref", type=Path, required=True)
    u.add_argument("--out", type=Path, required=True)
    u.add_argument("--version", required=True)
    u.add_argument("--clip", type=float, default=0.2)
    u.add_argument("--kl-coef", type=float, default=0.05)
    u.add_argument("--epochs", type=int, default=4)
    u.add_argument("--lr", type=float, default=1e-4)
    u.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.cmd == "build":
        result = build(args.rollout_dir, args.batch_dir, args.features, args.progress_weight)
    else:
        result = update(
            args.batch_dir,
            args.policy,
            args.ref,
            args.out,
            clip=args.clip,
            kl_coef=args.kl_coef,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            version=args.version,
        )
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
