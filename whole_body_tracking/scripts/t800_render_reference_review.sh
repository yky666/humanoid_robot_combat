#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_ROOT="${SRC_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
GMR_ROOT="${GMR_ROOT:-$SRC_ROOT/gmr}"
INPUT_ROOT="${INPUT_ROOT:-$REPO_ROOT/artifacts/boxing_T800}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/review_videos/t800_reference}"
PYTHON_BIN="${PYTHON_BIN:-/data2/yangky/miniconda3/envs/gmr/bin/python}"
RENDER_FPS="${RENDER_FPS:-25}"
MAX_SECONDS="${MAX_SECONDS:-8}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
VIEWS="${VIEWS:-front,right}"
OVERWRITE="${OVERWRITE:-1}"

default_inputs=(
  "$INPUT_ROOT/accad_combat_batch/accad_e1_jab_left_t800_tracking.npz"
  "$INPUT_ROOT/accad_combat_batch/accad_e2_jab_right_t800_tracking.npz"
  "$INPUT_ROOT/accad_combat_batch/accad_e5_hook_left_t800_tracking.npz"
  "$INPUT_ROOT/accad_combat_batch/accad_e6_hook_right_t800_tracking.npz"
  "$INPUT_ROOT/accad_combat_batch/accad_g3_front_kick_t800_tracking.npz"
  "$INPUT_ROOT/new_data/new_data_zhiquan_quanji_tracking.npz"
  "$INPUT_ROOT/new_data/new_data_540huixuantitui_tracking.npz"
)

if (( "$#" > 0 )); then
  inputs=("$@")
else
  inputs=("${default_inputs[@]}")
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[FAIL] missing python: $PYTHON_BIN"
  exit 1
fi
if [[ ! -d "$GMR_ROOT" ]]; then
  echo "[FAIL] missing GMR_ROOT: $GMR_ROOT"
  exit 1
fi

render_args=(
  --input-root "$INPUT_ROOT"
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
  if [[ ! -f "$input" ]]; then
    echo "[FAIL] missing input: $input"
    exit 1
  fi
  render_args+=(--input-file "$input")
done

mkdir -p "$OUTPUT_ROOT"
cd "$GMR_ROOT"
echo "[INFO] rendering ${#inputs[@]} T800 reference motion(s)"
echo "[INFO] output: $OUTPUT_ROOT"
exec env PYTHONPATH="$GMR_ROOT:${PYTHONPATH:-}" MUJOCO_GL="${MUJOCO_GL:-egl}" \
  "$PYTHON_BIN" scripts/render_motiondecode_t800_videos.py "${render_args[@]}"
