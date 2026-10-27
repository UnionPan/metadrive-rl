"""Per-run artifact contract (guideline §7) for the registered evaluations.

For each entry of ``configs/evaluation/experiments.yaml`` writes
``outputs/<id>/`` with: ``config.yaml`` (resolved host-driver config, wizard
config, run metadata, environment switches), ``git_commit.txt`` (the commit in
effect when the run started; runs predate per-run commit recording),
``environment.txt``, ``checkpoint_manifest.json`` (paths and SHA-256),
``rollout_policy_version.json`` (latent-policy bundles), ``metrics.json``
(scene-cluster bootstrap means), ``gpu_profile.json``, ``qualitative_notes.md``,
and symlinks ``rollout_logs/`` (native .asl) and ``videos/`` (AlpaSim MP4s).

Usage: python scripts/export_runs.py [--out outputs]
"""

import argparse
import datetime
import hashlib
import json
import subprocess
from functools import cache
from pathlib import Path

import torch
import yaml

from metadrive_ais.evaluation.summarize import METRICS, bootstrap_ci, rollouts, scene_means

REPO = Path(__file__).resolve().parents[1]


@cache
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


@cache
def environment() -> str:
    freeze = subprocess.run(
        ["uv", "pip", "freeze", "--python", str(Path.home() / "ext/alpasim/.venv/bin/python")],
        capture_output=True,
        text=True,
    ).stdout
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout
    return f"# GPUs\n{gpu}\n# AlpaSim venv (driver, evaluation)\n{freeze}"


def run_start(run: Path) -> datetime.datetime:
    first = (run / "driver.log").read_text().splitlines()[0]
    return datetime.datetime.strptime(first[1:20], "%Y-%m-%d %H:%M:%S")


def export(exp: dict, out_root: Path) -> None:
    run, out = REPO / exp["run"], out_root / exp["id"]
    out.mkdir(parents=True, exist_ok=True)
    driver_cfg = yaml.safe_load(next((run / "driver").glob("*-driver.yaml")).read_text())
    config = {
        "experiment": exp,
        "host_driver": driver_cfg,
        "wizard": yaml.safe_load((run / "sim/wizard-config.yaml").read_text()),
        "run_metadata": yaml.safe_load((run / "sim/run_metadata.yaml").read_text()),
    }
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    start = run_start(run)
    commit = subprocess.run(
        ["git", "log", "-1", f"--before={start:%Y-%m-%d %H:%M:%S}", "--format=%H %ad %s"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (out / "git_commit.txt").write_text(
        f"{commit}\n# last commit before the run started ({start}); uncommitted changes, if"
        " any, are not captured (runs predate per-run commit recording)\n"
    )
    (out / "environment.txt").write_text(environment())

    model = driver_cfg["model"]
    manifest = {"model_type": model["model_type"], "files": {}}
    ckpt = Path(model["checkpoint_path"])
    manifest["files"]["checkpoint"] = {"path": str(ckpt), "sha256": sha256(str(ckpt))}
    if model["model_type"] == "vavam_ais":
        bundle = torch.load(ckpt, map_location="cpu", weights_only=True)
        vavam = bundle["vavam_checkpoint"]
        manifest["files"]["vavam_checkpoint"] = {"path": vavam, "sha256": sha256(vavam)}
        (out / "rollout_policy_version.json").write_text(
            json.dumps(
                {
                    "policy_version": bundle.get("policy_version"),
                    "fit": bundle.get("fit"),
                    "latent": bundle.get("latent", {}).get("config"),
                },
                indent=1,
            )
        )
    tok = model.get("tokenizer_path")
    if tok:
        manifest["files"]["tokenizer"] = {"path": tok, "sha256": sha256(tok)}
    (out / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=1))

    rows, aborted = rollouts(run)
    scenes = sorted({r["scene"] for r in rows})
    metrics = {"scenes": len(scenes), "rollouts": len(rows), "aborted": aborted}
    for key, name in METRICS:
        values = scene_means(rows, key, scenes)
        lo, hi = bootstrap_ci(values)
        metrics[key] = {"name": name, "mean": float(values.mean()), "ci95": [lo, hi]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "gpu_profile.json").write_text(
        json.dumps({"note": "not recorded per run; measured costs in docs/HARDWARE.md"})
    )
    (out / "qualitative_notes.md").write_text(
        f"# {exp['label']}\n\nSplit: {exp['split']}. Discussion: `{exp['report']}`; "
        "failure modes: `reports/failure_taxonomy.md`.\n"
    )
    for name, target in (
        ("rollout_logs", run / "sim/rollouts"),
        ("videos", run / "sim/aggregate/videos/all"),
    ):
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        if target.exists():
            link.symlink_to(target)
    print(f"{exp['id']}: score {metrics['score']['mean']:.3f}, {len(rows)} rollouts -> {out}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=REPO / "outputs")
args = parser.parse_args()
for exp in yaml.safe_load((REPO / "configs/evaluation/experiments.yaml").read_text())[
    "experiments"
]:
    export(exp, args.out)
