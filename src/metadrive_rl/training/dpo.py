"""Direct preference optimisation of the policy head (Rafailov et al., 2023).

Preferences here are *simulator-derived*, not human: for each decision the
preferred action is the one whose plan best matches the logged expert future,
and rejected actions are candidates whose ADE is worse by at least
``--margin`` metres. The SFT checkpoint is both the initialisation and the
frozen reference policy.

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
    encoder = ckpt["encoder"]
    train = load_split(args.data, "train", encoder, device)
    val = load_split(args.data, "val", encoder, device)
    pairs = preference_pairs(train["ade"], args.margin, args.pairs_per_decision)
    with torch.no_grad():
        ref_logp = F.log_softmax(reference(train["features"], train["state"]), -1)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    log = open(args.out / "metrics.jsonl", "w")
    initial = {"val": evaluate(policy, val, args.tau)}
    log.write(json.dumps({"epoch": 0, **initial}) + "\n")
    best = (initial["val"]["regret_mean_m"], initial)
    start = time.time()
    for epoch in range(args.epochs):
        policy.train()
        perm = torch.randperm(len(pairs), device=device)
        for i in range(0, len(pairs), args.batch_size):
            d, w, lo = pairs[perm[i : i + args.batch_size]].T
            logp = F.log_softmax(policy(train["features"][d], train["state"][d]), -1)
            margin = (logp[torch.arange(len(d)), w] - ref_logp[d, w]) - (
                logp[torch.arange(len(d)), lo] - ref_logp[d, lo])
            loss = -F.logsigmoid(args.beta * margin).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 == 4 or epoch == args.epochs - 1:
            with torch.no_grad():
                d, w, lo = pairs.T
                logp = F.log_softmax(policy(train["features"], train["state"]), -1)
                pref_acc = float((logp[d, w] > logp[d, lo]).float().mean())
                kl = float((logp.exp() * (logp - ref_logp)).sum(-1).mean())
            row = {"epoch": epoch + 1, "pair_accuracy": pref_acc, "kl_to_ref": kl,
                   "val": evaluate(policy, val, args.tau)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if row["val"]["regret_mean_m"] < best[0]:
                best = (row["val"]["regret_mean_m"], row)
                save_checkpoint(args.out / "best.pt", policy, encoder,
                                f"dpo:{args.out.name}:e{epoch + 1}", {"metrics": row})
    summary = {"sft_val": initial["val"], "best": best[1], "pairs": len(pairs),
               "seconds": round(time.time() - start, 1)}
    if not (args.out / "best.pt").exists():  # DPO never beat the SFT start
        save_checkpoint(args.out / "best.pt", reference, encoder,
                        f"dpo:{args.out.name}:sft-init", {"metrics": initial})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
