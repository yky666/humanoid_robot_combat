#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-motiondecode_t800_render}"
GMR_ROOT="${GMR_ROOT:-/data2/yangky/test/gmr}"
PYTHON_BIN="${PYTHON_BIN:-/data2/yangky/miniconda3/envs/gmr/bin/python}"
RENDER_SCRIPT="${RENDER_SCRIPT:-$GMR_ROOT/scripts/render_motiondecode_t800_videos.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/yangky/test/datasets/MotionDecode_T800/videos}"
LOG_DIR="${LOG_DIR:-/data2/yangky/test/datasets/MotionDecode_T800/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/render_t800_$(date +%Y%m%d_%H%M%S).log}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-egl}"
WAIT_PID="${WAIT_PID:-}"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[ERROR] tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 1
fi

if [ "$#" -eq 0 ]; then
  set -- \
    --output-root "$OUTPUT_ROOT" \
    --render-fps 30 \
    --views front,right
fi

quote() {
  printf "%q" "$1"
}

quoted_args=""
for arg in "$@"; do
  quoted_args+=" $(quote "$arg")"
done

wait_block=""
if [ -n "$WAIT_PID" ]; then
  wait_block="
echo '[INFO] waiting for existing process PID $WAIT_PID to finish...'
while kill -0 $WAIT_PID 2>/dev/null; do
  date '+[INFO] %F %T existing render still running'
  sleep 60
done
echo '[INFO] existing process is gone, continuing render pass without overwrite'
"
fi

tmux new-session -d -s "$SESSION" "
set -euo pipefail
cd $(quote "$GMR_ROOT")
export MUJOCO_GL=$(quote "$MUJOCO_GL_BACKEND")
export PYTHONUNBUFFERED=1
exec > >(tee -a $(quote "$LOG_FILE")) 2>&1
echo '[INFO] session: $SESSION'
echo '[INFO] log: $LOG_FILE'
echo '[INFO] command: $PYTHON_BIN $RENDER_SCRIPT$quoted_args'
$wait_block
$(quote "$PYTHON_BIN") $(quote "$RENDER_SCRIPT")$quoted_args
status=\$?
echo \"[INFO] render exited with status \${status}\"
echo '[INFO] press Ctrl-b d to detach, or exit to close this shell'
exec bash
"

echo "[OK] started tmux session: $SESSION"
echo "[OK] log: $LOG_FILE"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b then d"
