#!/usr/bin/env bash
# M1 controllability probe: drive the same scene once per fixed high-level
# action and keep each run's logs, metrics and video under <out-dir>/<label>.
#
# Usage: scripts/mode_probe.sh <out-dir> [policy-spec...]
# Env:   DRIVER_SOURCE (default metadrive_no_render), plus run_episode.sh's.
set -uo pipefail

OUT=$(realpath -m "${1:?usage: mode_probe.sh <out-dir> [policy-spec...]}")
shift
SPECS=("$@")
if [[ ${#SPECS[@]} -eq 0 ]]; then
  SPECS=(rule fixed:CRUISE@10 fixed:FOLLOW fixed:YIELD fixed:BRAKE
         fixed:CHANGE_LANE_LEFT@10 fixed:CHANGE_LANE_RIGHT@10)
fi
export DRIVER_SOURCE=${DRIVER_SOURCE:-metadrive_no_render}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUT"
for spec in "${SPECS[@]}"; do
  label=$(echo "$spec" | tr ':@' '__')
  echo "=== $spec -> $OUT/$label"
  rm -rf "${OUT:?}/$label"
  "$HERE/run_episode.sh" "$OUT/$label" "$spec" > "$OUT/$label.log" 2>&1
  echo "exit=$?"
done
