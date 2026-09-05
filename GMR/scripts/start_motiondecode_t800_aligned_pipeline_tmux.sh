#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-motiondecode_t800_aligned_pipeline}"
GMR_ROOT="${GMR_ROOT:-/data2/yangky/test/gmr}"
PYTHON_BIN="${PYTHON_BIN:-/data2/yangky/miniconda3/envs/gmr/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/yangky/test/datasets/MotionDecode_T800_aligned}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-egl}"
RENDER_FPS="${RENDER_FPS:-30}"
VIEWS="${VIEWS:-front,right}"
MAX_SECONDS="${MAX_SECONDS:-}"
MAX_FILES="${MAX_FILES:-}"
RENDER_LIMIT="${RENDER_LIMIT:-}"
NICE_LEVEL="${NICE_LEVEL:-5}"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[ERROR] tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

quote() {
  printf "%q" "$1"
}

join_quoted() {
  local result=""
  for arg in "$@"; do
    result+=" $(quote "$arg")"
  done
  printf "%s" "$result"
}

convert_args=(
  --output-root "$OUTPUT_ROOT"
  --write-csv
  --align-floor
  --force
)
if [ -n "$MAX_FILES" ]; then
  convert_args+=(--max-files "$MAX_FILES")
fi

render_args=(
  --input-root "$OUTPUT_ROOT/gmr_pkl"
  --manifest "$OUTPUT_ROOT/manifest.json"
  --output-root "$OUTPUT_ROOT/videos_mesh_fixed"
  --robot-xml "$GMR_ROOT/assets/t800/t800_visual.xml"
  --render-fps "$RENDER_FPS"
  --views "$VIEWS"
  --overwrite
)
if [ -n "$MAX_SECONDS" ]; then
  render_args+=(--max-seconds "$MAX_SECONDS")
fi
if [ -n "$RENDER_LIMIT" ]; then
  render_args+=(--limit "$RENDER_LIMIT")
fi

build_cmd="nice -n $(quote "$NICE_LEVEL") $(quote "$PYTHON_BIN") $(quote "$GMR_ROOT/scripts/build_t800_visual_mjcf.py")"
convert_cmd="nice -n $(quote "$NICE_LEVEL") $(quote "$PYTHON_BIN") $(quote "$GMR_ROOT/scripts/motiondecode_g1_csv_to_t800.py")$(join_quoted "${convert_args[@]}")"
render_cmd="nice -n $(quote "$NICE_LEVEL") $(quote "$PYTHON_BIN") $(quote "$GMR_ROOT/scripts/render_motiondecode_t800_videos.py")$(join_quoted "${render_args[@]}")"

tmux new-session -d -s "$SESSION" "
set -euo pipefail
cd $(quote "$GMR_ROOT")
export MUJOCO_GL=$(quote "$MUJOCO_GL_BACKEND")
export PYTHONUNBUFFERED=1
exec > >(tee -a $(quote "$LOG_FILE")) 2>&1
echo '[INFO] session: $SESSION'
echo '[INFO] output root: $OUTPUT_ROOT'
echo '[INFO] log: $LOG_FILE'
echo '[INFO] build visual MJCF'
$build_cmd
echo '[INFO] convert G1 CSV -> T800 aligned data'
$convert_cmd
echo '[INFO] render aligned T800 mesh videos'
$render_cmd
echo '[OK] pipeline complete'
echo '[INFO] press Ctrl-b d to detach, or exit to close this shell'
exec bash
"

echo "[OK] started tmux session: $SESSION"
echo "[OK] output root: $OUTPUT_ROOT"
echo "[OK] log: $LOG_FILE"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b then d"
