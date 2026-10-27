"""Per-iteration table of an M5a RL loop (``<out>/bundle_v<k>.stats.json``).

Mean score is on that iteration's own scene draw (different scenes each
iteration), so it is a diagnostic, not an evaluation.

Usage: python scripts/rl_progress.py runs/rl/v3
"""

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
COLUMNS = ["iter", "rollouts", "aborted", "mean score", "signal scenes", "queries"]
COLUMNS += ["PPO epochs", "approx KL", "KL to v0", "prior std"]
print("| " + " | ".join(COLUMNS) + " |")
print("|" + "---|" * len(COLUMNS))
for k in range(100):
    f = out / f"bundle_v{k + 1}.stats.json"
    if not f.exists():
        break
    s = json.loads(f.read_text())
    h = s["history"][-1]
    signal = s["scenes"] - s["degenerate_scenes"]
    print(
        f"| {k} | {s['rollouts_scored']} | {s['rollouts_aborted']} | {s['mean_score']:.3f} | "
        f"{signal}/{s['scenes']} | {s['queries']} | {h['epoch']} | {h['approx_kl']:.4f} | "
        f"{h['kl_ref']:.3f} | {h['prior_std_mean']:.2f} |"
    )
