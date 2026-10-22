"""Direct preference optimisation of the policy head (Rafailov et al., 2023).

Preferences are *simulator-derived*, not human. Two sources:

* default: per expert decision, the lowest-ADE candidate is preferred over
  candidates at least ``--margin`` metres worse (same information as SFT's
  targets; kept as a baseline);
* ``--pairs``: matched-state branch outcomes from
  ``metadrive_rl.data.branch_prefs`` (safety, then progress), with
  ``--val-pairs`` from held-out scenes for preference accuracy.

The SFT checkpoint is both the initialisation and the frozen reference policy.

    L = -log sigmoid(beta * [(log pi(w) - log pi_ref(w)) - (log pi(l) - log pi_ref(l))])

Usage:
    python -m metadrive_rl.training.dpo --data data/datasets/expert_v1 \\
        --sft runs/sft/<name>/best.pt --out runs/dpo/<name>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from metadrive_rl.policy.neural import PolicyHead, save_checkpoint

from .sft import evaluate, load_split


def preference_pairs(ade: torch.Tensor, margin: float, max_per_decision: int) -> torch.Tensor:
    """(decision, winner, loser) index triples; winner = argmin ADE."""
    winner = ade.argmin(-1)
    gap = ade - ade.min(-1, keepdim=True).values
    triples = []
    for i in range(len(ade)):
        losers = torch.nonzero(gap[i] >= margin)[:, 0]
        # Hardest negatives first: the rejected actions closest to the winner.
        losers = losers[torch.argsort(gap[i, losers])][:max_per_decision]
        triples += [(i, int(winner[i]), int(j)) for j in losers]
    return torch.tensor(triples, dtype=torch.long, device=ade.device)


def load_branch_pairs(pair_dir: Path, device: torch.device) -> dict:
    states = pl.read_parquet(pair_dir / "states.parquet")
    row = {s: i for i, s in enumerate(states["scene_id"].to_list())}
    pairs = pl.read_parquet(pair_dir / "pairs.parquet")
    return {
        "features": torch.tensor(np.load(pair_dir / "features.npy"), device=device),
        "state": torch.tensor(np.array(states["state"].to_list()), dtype=torch.float32, device=device),
        "pairs": torch.tensor([[row[r["scene_id"]], r["winner"], r["loser"]] for r in pairs.to_dicts()],
                              dtype=torch.long, device=device),
    }


@torch.no_grad()
def pair_accuracy(head: PolicyHead, data: dict) -> float:
    d, w, lo = data["pairs"].T
    logp = F.log_softmax(head(data["features"], data["state"]), -1)
    return float((logp[d, w] > logp[d, lo]).float().mean())


def load_head(path: Path, device: torch.device) -> tuple[PolicyHead, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    head = PolicyHead(**ckpt["head_config"]).to(device)
    head.load_state_dict(ckpt["state_dict"])
    return head, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, help="branch preference dir (train scenes)")
    parser.add_argument("--val-pairs", type=Path, help="branch preference dir (val scenes)")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=1.0, help="min ADE gap (m) for a pair")
    parser.add_argument("--pairs-per-decision", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.5, help="only for reported val loss")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ckpt = load_head(args.sft, device)
    reference, _ = load_head(args.sft, device)
    reference.eval().requires_grad_(False)
    encoder, context = ckpt["encoder"], ckpt.get("context", 1)
    train = load_split(args.data, "train", encoder, device, context)
    val = load_split(args.data, "val", encoder, device, context)
    if args.pairs:
        # Branch preferences: decisions index branch states, not expert rows.
        pref = load_branch_pairs(args.pairs, device)
        val_pref = load_branch_pairs(args.val_pairs, device) if args.val_pairs else None
    else:
        pref = {"features": train["features"], "state": train["state"],
                "pairs": preference_pairs(train["ade"], args.margin, args.pairs_per_decision)}
        val_pref = None
    pairs = pref["pairs"]
    with torch.no_grad():
        ref_logp = F.log_softmax(reference(pref["features"], pref["state"]), -1)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    log = open(args.out / "metrics.jsonl", "w")
    initial = {"val": evaluate(policy, val, args.tau)}
    if args.pairs:
        initial["pair_accuracy"] = pair_accuracy(policy, pref)
        if val_pref is not None:
            initial["val_pair_accuracy"] = pair_accuracy(policy, val_pref)
    log.write(json.dumps({"epoch": 0, **initial}) + "\n")
    def selection_loss(row: dict) -> float:
        # Branch mode optimises held-out preferences; otherwise imitation regret.
        if val_pref is not None:
            return -row["val_pair_accuracy"]
        return row["val"]["regret_mean_m"]

    best = (selection_loss(initial), initial)
    start = time.time()
    for epoch in range(args.epochs):
        policy.train()
        perm = torch.randperm(len(pairs), device=device)
        for i in range(0, len(pairs), args.batch_size):
            d, w, lo = pairs[perm[i : i + args.batch_size]].T
            logp = F.log_softmax(policy(pref["features"][d], pref["state"][d]), -1)
            margin = (logp[torch.arange(len(d)), w] - ref_logp[d, w]) - (
                logp[torch.arange(len(d)), lo] - ref_logp[d, lo])
            loss = -F.logsigmoid(args.beta * margin).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 == 4 or epoch == args.epochs - 1:
            with torch.no_grad():
                d, w, lo = pairs.T
                logp = F.log_softmax(policy(pref["features"], pref["state"]), -1)
                pref_acc = float((logp[d, w] > logp[d, lo]).float().mean())
                kl = float((logp.exp() * (logp - ref_logp)).sum(-1).mean())
            row = {"epoch": epoch + 1, "pair_accuracy": pref_acc, "kl_to_ref": kl,
                   "val": evaluate(policy, val, args.tau)}
            if val_pref is not None:
                row["val_pair_accuracy"] = pair_accuracy(policy, val_pref)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if selection_loss(row) < best[0]:
                best = (selection_loss(row), row)
                save_checkpoint(args.out / "best.pt", policy, encoder,
                                f"dpo:{args.out.name}:e{epoch + 1}",
                                {"metrics": row, "context": context})
    summary = {"sft_val": initial["val"], "best": best[1], "pairs": len(pairs),
               "seconds": round(time.time() - start, 1)}
    if not (args.out / "best.pt").exists():  # DPO never beat the SFT start
        save_checkpoint(args.out / "best.pt", reference, encoder,
                        f"dpo:{args.out.name}:sft-init", {"metrics": initial, "context": context})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
