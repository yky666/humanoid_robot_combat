#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-motiondecode_t800_train}"
REPO_ROOT="${REPO_ROOT:-/data2/yangky/test/whole_body_tracking}"
MOTION_FILE="${MOTION_FILE:-/data2/yangky/test/datasets/MotionDecode_T800_aligned/tracking_manifest_csv.json}"
LOG_DIR="${LOG_DIR:-/data2/yangky/test/datasets/MotionDecode_T800_aligned/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_motiondecode_t800_$(date +%Y%m%d_%H%M%S).log}"
PYTHON_BIN="${PYTHON_BIN:-/data2/yangky/miniconda3/envs/env_isaaclab/bin/python}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-128}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
RUN_NAME="${RUN_NAME:-motiondecode_t800_all}"
LOGGER="${LOGGER:-tensorboard}"
PROJECT="${PROJECT:-t800_motiondecode}"
TASK_VARIANT="${TASK_VARIANT:-default}"
TMPDIR_ROOT="${TMPDIR_ROOT:-/data2/yangky/test/tmp}"

quote() {
  printf "%q" "$1"
}

join_quoted() {
  local out=""
  local arg
  for arg in "$@"; do
    out+=" $(quote "$arg")"
  done
  printf "%s" "$out"
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[ERROR] tmux session already exists: $SESSION"
  echo "        Attach with: tmux attach -t $SESSION"
  exit 1
fi

mkdir -p "$LOG_DIR" "$TMPDIR_ROOT"

train_args=(
  --motion_file "$MOTION_FILE"
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERATIONS"
  --device "$DEVICE"
  --run_name "$RUN_NAME"
  --logger "$LOGGER"
  --log_project_name "$PROJECT"
  --task_variant "$TASK_VARIANT"
  --headless
  "$@"
)

cmd="set -euo pipefail"
cmd+="; mkdir -p $(quote "$LOG_DIR") $(quote "$TMPDIR_ROOT")"
cmd+="; cd $(quote "$REPO_ROOT")"
cmd+="; export PYTHONUNBUFFERED=1"
cmd+="; export CUDA_VISIBLE_DEVICES=$(quote "$GPU")"
cmd+="; export TMPDIR=$(quote "$TMPDIR_ROOT")"
cmd+="; export OMNI_USER_CACHE_DIR=$(quote "$TMPDIR_ROOT/omni_cache")"
cmd+="; export OV_CACHE_ROOT=$(quote "$TMPDIR_ROOT/ov_cache")"
cmd+="; export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cmd+="; set +e"
cmd+="; $(quote "$PYTHON_BIN") $(quote "$REPO_ROOT/scripts/rsl_rl/train_t800.py")$(join_quoted "${train_args[@]}") 2>&1 | tee -a $(quote "$LOG_FILE")"
cmd+="; status=\${PIPESTATUS[0]}"
cmd+="; echo [INFO] train exited with status \$status"
cmd+="; echo [INFO] log: $(quote "$LOG_FILE")"
cmd+="; echo [INFO] press Ctrl-b d to detach, or exit to close this shell"
cmd+="; exit \$status"

tmux new-session -d -s "$SESSION" "bash -lc $(quote "$cmd")"

echo "[OK] started tmux session: $SESSION"
echo "[OK] log: $LOG_FILE"
echo "[OK] attach: tmux attach -t $SESSION"
