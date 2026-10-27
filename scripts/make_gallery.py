"""Side-by-side rollout gallery (guideline M6): the same scenes under several policies.

For every scene that all runs completed, takes each run's first rollout of that
scene (by AlpaSim's results order), stacks the simulator's own MP4s left to right
(no re-rendering), and writes ``<out>/<scene>.mp4`` plus ``<out>/index.md`` with
labels, outcomes and SHA-256 checksums. Rollouts of different runs are
independent: CATK traffic is stochastic, so the same scene is not the same
traffic realisation.

Usage:
    python scripts/make_gallery.py --out outputs/demo_gallery \\
        "B0b* (stock VaVAM-B)=runs/vavam/m1_mup/b0b" "AIS adapter=runs/ais/m3_cl/frame_s0" ...
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("runs", nargs="+", help="label=run_dir")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

runs = [spec.rsplit("=", 1) for spec in args.runs]
first = {}  # label -> scene -> rollout record
for label, run in runs:
    rollouts = json.loads(Path(run, "sim/aggregate/results-summary.json").read_text())["rollouts"]
    first[label] = {}
    for r in rollouts:
        if "error" not in r["metrics"]:
            first[label].setdefault(r["clipgt_id"], (Path(run), r))
scenes = sorted(set.intersection(*(set(v) for v in first.values())))

args.out.mkdir(parents=True, exist_ok=True)
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
labels = [label for label, _ in runs]
lines = [
    "# Demo gallery",
    "",
    "Left to right: " + " | ".join(f"**{label}** (`{run}`)" for label, run in runs) + ".",
    "Each panel is AlpaSim's own rollout video (front camera and map view) of that run's",
    "first rollout of the scene. Rollouts are independent across panels (stochastic",
    "CATK traffic). Outcome: score (0 on failure, else clipped progress) and failure reason.",
    "",
    "| scene | " + " | ".join(labels) + " | video | sha256 |",
    "|---|" + "---|" * (len(labels) + 2),
]
for scene in scenes:
    videos, cells = [], []
    for label in labels:
        run, r = first[label][scene]
        video = next((run / "sim/rollouts" / scene / r["rollout_id"]).glob("*.mp4"))
        videos.append(video)
        cells.append(f"{r['score']:.2f} {r.get('failure_reason') or 'pass'}")
    out = args.out / f"{scene}.mp4"
    inputs = [x for v in videos for x in ("-i", str(v))]
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            *inputs,
            "-filter_complex",
            f"hstack=inputs={len(videos)}:shortest=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "28",
            str(out),
        ],
        check=True,
    )
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    lines.append(f"| `{scene}` | " + " | ".join(cells) + f" | `{out.name}` | `{digest}` |")
    print(scene, cells)
(args.out / "index.md").write_text("\n".join(lines) + "\n")
print(f"{len(scenes)} scenes -> {args.out}/index.md")
