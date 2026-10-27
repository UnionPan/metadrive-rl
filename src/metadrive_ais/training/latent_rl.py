"""Policy-gradient update of the latent prior q_phi(c | z_t) from AlpaSim rollouts (M5a).

The policy is the stochastic latent action of the VaVAM-AIS driver
(:mod:`metadrive_ais.models.latent_policy`). The memory, adapter, latent
decoder and VaVAM stay frozen, so the logged AIS state ``z_t`` is exact and
``log q_phi(c_t | z_t)`` can be recomputed for the update. VaVAM's flow noise is
an environment input; its density is not optimised.

- Reward per rollout: AlpaSim's score (0 when the rollout fails, otherwise the
  clipped relative progress), the evaluation metric.
- Advantage: score minus the mean score of the same scene's rollouts in the
  batch (group baseline), divided by the batch standard deviation of those
  differences. Scenes whose rollouts all score the same contribute nothing;
  they are counted.
- Credit: every latent decision whose effect falls after the forced replay and
  up to the first incident (offroad or any collision), where AlpaSim stops
  scoring. With a held latent (``hold`` = k queries), only decision queries
  carry a log-probability; a decision counts if its hold window
  [t, t + (k - 1) * 0.5 s] overlaps that interval.
- Update: clipped PPO on the logged (z_t, c_t, log q_old), plus
  ``ref_weight * KL(q_phi || q_ref)`` to keep latents where the decoder was
  trained (q_ref: the prior before RL).
- Checks: logged log-probabilities match the recomputed ones; every record
  carries the bundle's policy version.

Usage:
    python -m metadrive_ais.training.latent_rl --bundle <bundle.pt> --runs <run-dir> ... \\
        --logs <latent.jsonl> ... --out <next-bundle.pt> [--reference <bundle-v0.pt>]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch

from metadrive_ais.data.asl import find_rollouts, iter_entries
from metadrive_ais.models.latent_policy import LOG_STD_RANGE, LatentChannel

INCIDENT_METRICS = ("offroad", "collision_any")
QUERY_PERIOD_US = 500_000  # VaVAM is queried at 2 Hz


def rollout_windows(run_dir: Path) -> dict[str, tuple[int, int | None]]:
    """session uuid -> (control start, first incident or None), in microseconds."""
    out = {}
    for path in find_rollouts(run_dir):
        for e in iter_entries(path):
            if e.WhichOneof("log_entry") == "rollout_metadata":
                m = e.rollout_metadata
                start = int(m.session_metadata.start_timestamp_us) + int(m.force_gt_duration)
                session = m.session_metadata.session_uuid
                break
        else:
            continue
        incident = None
        metrics_file = path.parent / "metrics.parquet"
        if metrics_file.exists():
            hits = pl.read_parquet(metrics_file).filter(
                pl.col("name").is_in(INCIDENT_METRICS) & pl.col("valid") & (pl.col("values") > 0)
            )
            if len(hits):
                incident = int(hits["timestamps_us"].min())
        out[session] = (start, incident)
    return out


def rollout_scores(run_dir: Path) -> tuple[dict[str, tuple[str, float]], int]:
    """session uuid -> (scene, score); and the number of simulator-aborted rollouts."""
    out, aborted = {}, 0
    for f in sorted(run_dir.glob("**/aggregate/results-summary.json")):
        for r in json.loads(f.read_text())["rollouts"]:
            if "error" in r["metrics"]:
                aborted += 1
                continue
            out[r["rollout_id"]] = (r["clipgt_id"], float(r["score"]))
    return out, aborted


def advantages(scores: dict[str, tuple[str, float]]) -> tuple[dict[str, float], dict]:
    groups = defaultdict(list)
    for session, (scene, score) in scores.items():
        groups[scene].append((session, score))
    centred = {}
    degenerate = 0
    for members in groups.values():
        values = np.array([s for _, s in members])
        if len(members) < 2 or np.ptp(values) == 0:
            degenerate += 1
            continue
        for session, score in members:
            centred[session] = score - values.mean()
    scale = np.std(list(centred.values())) if centred else 1.0
    adv = {k: v / (scale + 1e-8) for k, v in centred.items()}
    stats = {"scenes": len(groups), "degenerate_scenes": degenerate, "advantage_scale": scale}
    return adv, stats


def load_queries(logs, windows, adv, policy_version):
    z, c, logp, a, sessions = [], [], [], [], set()
    skipped = defaultdict(int)
    for log in logs:
        for line in Path(log).read_text().splitlines():
            r = json.loads(line)
            if r["policy_version"] != policy_version:
                raise ValueError(f"{log}: record of policy version {r['policy_version']}")
            if r["mode"] != "sample":
                raise ValueError(f"{log}: latents were not sampled ({r['mode']})")
            session = r["session"]
            if session not in adv:
                skipped["no advantage"] += 1
                continue
            if not r.get("decision", True):
                skipped["held latent"] += 1
                continue
            start, incident = windows[session]
            if r["time_us"] + (r.get("hold", 1) - 1) * QUERY_PERIOD_US < start:
                skipped["forced replay"] += 1
                continue
            if incident is not None and r["time_us"] > incident:
                skipped["after incident"] += 1
                continue
            z.append(r["z"]), c.append(r["c"]), logp.append(r["logp"]), a.append(adv[session])
            sessions.add(session)
    tensors = [torch.tensor(x, dtype=torch.float32) for x in (z, c, logp, a)]
    return tensors, {"queries": len(z), "rollouts": len(sessions), "skipped": dict(skipped)}


def ppo_update(
    latent: LatentChannel,
    reference: LatentChannel,
    z: torch.Tensor,
    c: torch.Tensor,
    logp_old: torch.Tensor,
    a: torch.Tensor,
    epochs: int = 4,
    minibatch: int = 256,
    lr: float = 3e-5,
    clip: float = 0.2,
    ref_weight: float = 0.1,
    target_kl: float = 0.02,
) -> list[dict]:
    """Clipped PPO on the prior's log-probabilities; returns per-epoch diagnostics.

    Stops after an epoch whose approximate KL(old || new) exceeds ``target_kl``.
    """
    opt = torch.optim.Adam(latent.prior.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        order = torch.randperm(len(z))
        for k in range(0, len(z), minibatch):
            idx = order[k : k + minibatch]
            dist = latent.prior(z[idx])
            ratio = (dist.log_prob(c[idx]).sum(-1) - logp_old[idx]).exp()
            clipped = ratio.clamp(1 - clip, 1 + clip)
            pg = -torch.min(ratio * a[idx], clipped * a[idx]).mean()
            kl_ref = torch.distributions.kl_divergence(dist, reference.prior(z[idx])).sum(-1).mean()
            loss = pg + ref_weight * kl_ref
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            dist = latent.prior(z)
            logp_new = dist.log_prob(c).sum(-1)
            ratio = (logp_new - logp_old).exp()
            approx_kl = float(((ratio - 1) - ratio.log()).mean())  # >= 0, low variance
            history.append(
                {
                    "epoch": epoch + 1,
                    "approx_kl": approx_kl,
                    "ratio_mean": float(ratio.mean()),
                    "clip_fraction": float(((ratio - 1).abs() > clip).float().mean()),
                    "kl_ref": float(
                        torch.distributions.kl_divergence(dist, reference.prior(z)).sum(-1).mean()
                    ),
                    "prior_std_mean": float(dist.stddev.mean()),
                    # Hard-clamped log-std: no gradient for these dimensions.
                    "prior_std_at_clamp": float(
                        (dist.stddev.log() >= LOG_STD_RANGE[1] - 1e-6).float().mean()
                    ),
                    "surrogate": float((ratio * a).mean()),
                }
            )
        if approx_kl > target_kl:
            break
    return history


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="prior before RL (default: --bundle)")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--logs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    # z_t is unnormalised: one Adam step at 3e-4 moved the prior by KL ~0.1 (smoke test).
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ref-weight", type=float, default=0.1)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    version = int(bundle.get("policy_version", 0))
    latent = LatentChannel(**bundle["latent"]["config"])
    latent.load_state_dict(bundle["latent"]["state_dict"])
    reference = LatentChannel(**bundle["latent"]["config"])
    ref_bundle = torch.load(args.reference or args.bundle, map_location="cpu", weights_only=True)
    reference.load_state_dict(ref_bundle["latent"]["state_dict"])
    reference.requires_grad_(False)

    windows, scores, aborted = {}, {}, 0
    for run in args.runs:
        windows.update(rollout_windows(run))
        s, n = rollout_scores(run)
        scores.update(s)
        aborted += n
    adv, adv_stats = advantages(scores)
    (z, c, logp_old, a), query_stats = load_queries(args.logs, windows, adv, version)
    if not len(z):
        raise SystemExit("no usable queries")

    with torch.no_grad():
        recomputed = latent.prior(z).log_prob(c).sum(-1)
    logp_error = float((recomputed - logp_old).abs().max())
    if logp_error > 1e-3:
        raise SystemExit(f"logged log-probs do not match the bundle's prior ({logp_error:.2e})")

    history = ppo_update(
        latent,
        reference,
        z,
        c,
        logp_old,
        a,
        epochs=args.epochs,
        minibatch=args.minibatch,
        lr=args.lr,
        clip=args.clip,
        ref_weight=args.ref_weight,
        target_kl=args.target_kl,
    )

    bundle["latent"]["state_dict"] = latent.state_dict()
    bundle["policy_version"] = version + 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.out)
    score_values = [s for _, s in scores.values()]
    stats = {
        "policy_version_in": version,
        "policy_version_out": version + 1,
        "rollouts_scored": len(scores),
        "rollouts_aborted": aborted,
        "mean_score": float(np.mean(score_values)),
        **adv_stats,
        **query_stats,
        "logp_max_error": logp_error,
        "history": history,
    }
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "history"}))
    print(json.dumps(history[-1]))


if __name__ == "__main__":
    main()
