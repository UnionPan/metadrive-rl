"""Advantage-weighted regression of the latent prior on logged RL rollouts (M5a).

The equal-budget non-RL comparator for latent-policy RL: the same rollouts
(every iteration of a ``scripts/latent_rl_loop.sh`` run), the same advantages
(per iteration, scene-group baseline, see :mod:`metadrive_ais.training.latent_rl`),
but one offline fit of q_phi(c | z) to the logged latents weighted by
``exp(A / beta)``. This is reward-weighted imitation, not RL: no on-policy
updates, no importance ratios. The fit starts from the v0 prior.

Usage:
    python -m metadrive_ais.training.latent_awr --loop runs/rl/v3b --out <bundle.pt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from metadrive_ais.models.latent_policy import LatentChannel
from metadrive_ais.training.latent_rl import (
    advantages,
    load_queries,
    rollout_scores,
    rollout_windows,
)


def iteration_queries(it: Path):
    """(z, c, advantage) of one loop iteration, any policy version."""
    windows, scores = {}, {}
    for run in sorted(p for p in it.glob("stack_*") if p.is_dir()):
        windows.update(rollout_windows(run))
        scores.update(rollout_scores(run)[0])
    adv, _ = advantages(scores)
    version = json.loads(sorted(it.glob("latent_*.jsonl"))[0].open().readline())["policy_version"]
    (z, c, _, a), stats = load_queries(sorted(it.glob("latent_*.jsonl")), windows, adv, version)
    return z, c, a, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--loop", type=Path, required=True, help="latent_rl_loop.sh output dir")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=300, help="full-batch Adam steps")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    bundle = torch.load(args.loop / "bundle_v0.pt", map_location="cpu", weights_only=True)
    latent = LatentChannel(**bundle["latent"]["config"])
    latent.load_state_dict(bundle["latent"]["state_dict"])

    parts = [iteration_queries(it) for it in sorted(args.loop.glob("iter_*"))]
    z, c, a = (torch.cat([p[i] for p in parts]) for i in range(3))
    w = (a / args.beta).exp().clamp(max=args.max_weight)
    w = w / w.mean()

    opt = torch.optim.Adam(latent.prior.parameters(), lr=args.lr)
    for _ in range(args.steps):
        loss = -(w * latent.prior(z).log_prob(c).sum(-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        dist = latent.prior(z)
        stats = {
            "iterations": len(parts),
            "queries": len(z),
            "weight_max": float(w.max()),
            "weighted_nll": float(loss),
            "prior_mean_abs": float(dist.mean.abs().mean()),
            "prior_std_mean": float(dist.stddev.mean()),
        }
    bundle["latent"]["state_dict"] = latent.state_dict()
    bundle["policy_version"] = 0  # the driver logs an int version; the origin is in "fit"
    bundle["fit"] = f"awr on {args.loop} (beta {args.beta})"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.out)
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
