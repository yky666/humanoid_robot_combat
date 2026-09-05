#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_ROOT="${SRC_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
GMR_ROOT="${GMR_ROOT:-$SRC_ROOT/gmr}"
PYTHON_BIN="${PYTHON_BIN:-/data2/yangky/miniconda3/envs/gmr/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/review_videos/t800_reference_audit_20260902/punching}"
MOTIONDECODE_PUNCH_ROOT="${MOTIONDECODE_PUNCH_ROOT:-$SRC_ROOT/datasets/MotionDecode_T800_aligned/tracking_npz/samples/4.Martial_Arts/Punching_Techniques}"
ACCAD_BOXING_ROOT="${ACCAD_BOXING_ROOT:-$REPO_ROOT/artifacts/boxing_T800/accad_combat_batch}"
RENDER_FPS="${RENDER_FPS:-25}"
MAX_SECONDS="${MAX_SECONDS:-10}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
VIEWS="${VIEWS:-front,right}"
OVERWRITE="${OVERWRITE:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[FAIL] missing python: $PYTHON_BIN"
  exit 1
fi
if [[ ! -d "$GMR_ROOT" ]]; then
  echo "[FAIL] missing GMR_ROOT: $GMR_ROOT"
  exit 1
fi
if [[ ! -d "$MOTIONDECODE_PUNCH_ROOT" ]]; then
  echo "[FAIL] missing MotionDecode punching root: $MOTIONDECODE_PUNCH_ROOT"
  exit 1
fi

inputs=(
  "$ACCAD_BOXING_ROOT/accad_e1_jab_left_t800_tracking.npz"
  "$ACCAD_BOXING_ROOT/accad_e2_jab_right_t800_tracking.npz"
  "$ACCAD_BOXING_ROOT/accad_e5_hook_left_t800_tracking.npz"
  "$ACCAD_BOXING_ROOT/accad_e6_hook_right_t800_tracking.npz"
)

while IFS= read -r motion_path; do
  inputs+=("$motion_path")
done < <(find "$MOTIONDECODE_PUNCH_ROOT" -type f -name '*_t800_tracking.npz' | sort)

for input in "${inputs[@]}"; do
  if [[ ! -f "$input" ]]; then
    echo "[FAIL] missing input: $input"
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"
printf '%s\n' "${inputs[@]}" > "$OUTPUT_ROOT/candidate_inputs.txt"

render_args=(
  --input-root /tmp/t800_reference_audit_flat
  --input-format auto
  --output-root "$OUTPUT_ROOT"
  --render-fps "$RENDER_FPS"
  --max-seconds "$MAX_SECONDS"
  --width "$WIDTH"
  --height "$HEIGHT"
  --views "$VIEWS"
  --index-file "$OUTPUT_ROOT/index.html"
)
if [[ "$OVERWRITE" == "1" ]]; then
  render_args+=(--overwrite)
fi
for input in "${inputs[@]}"; do
  render_args+=(--input-file "$input")
done

cd "$GMR_ROOT"
echo "[INFO] rendering ${#inputs[@]} T800 punching reference candidate(s)"
echo "[INFO] output: $OUTPUT_ROOT"
exec env PYTHONPATH="$GMR_ROOT:${PYTHONPATH:-}" MUJOCO_GL="${MUJOCO_GL:-egl}" \
  "$PYTHON_BIN" scripts/render_motiondecode_t800_videos.py "${render_args[@]}"
