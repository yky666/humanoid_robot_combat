#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-motiondecode_t800_npz}"
REPO_ROOT="${REPO_ROOT:-/data2/yangky/test/whole_body_tracking}"
MANIFEST="${MANIFEST:-/data2/yangky/test/datasets/MotionDecode_T800_aligned/tracking_manifest.json}"
LOG_DIR="${LOG_DIR:-/data2/yangky/test/datasets/MotionDecode_T800_aligned/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/npz_prepare_$(date +%Y%m%d_%H%M%S).log}"
PYTHON_BIN="${PYTHON_BIN:-/data2/yangky/miniconda3/envs/env_isaaclab/bin/python}"
GPU="${GPU:-0}"
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

prepare_args=(--manifest "$MANIFEST" "$@")

cmd="set -euo pipefail"
cmd+="; mkdir -p $(quote "$LOG_DIR") $(quote "$TMPDIR_ROOT")"
cmd+="; cd $(quote "$REPO_ROOT")"
cmd+="; export PYTHONUNBUFFERED=1"
cmd+="; export CUDA_VISIBLE_DEVICES=$(quote "$GPU")"
cmd+="; export TMPDIR=$(quote "$TMPDIR_ROOT")"
cmd+="; export OMNI_USER_CACHE_DIR=$(quote "$TMPDIR_ROOT/omni_cache")"
cmd+="; export OV_CACHE_ROOT=$(quote "$TMPDIR_ROOT/ov_cache")"
cmd+="; set +e"
cmd+="; $(quote "$PYTHON_BIN") $(quote "$REPO_ROOT/scripts/batch_prepare_t800_motions.py")$(join_quoted "${prepare_args[@]}") 2>&1 | tee -a $(quote "$LOG_FILE")"
cmd+="; status=\${PIPESTATUS[0]}"
cmd+="; echo [INFO] npz conversion exited with status \$status"
cmd+="; echo [INFO] log: $(quote "$LOG_FILE")"
cmd+="; echo [INFO] press Ctrl-b d to detach, or exit to close this shell"
cmd+="; exit \$status"

tmux new-session -d -s "$SESSION" "bash -lc $(quote "$cmd")"

echo "[OK] started tmux session: $SESSION"
echo "[OK] log: $LOG_FILE"
echo "[OK] attach: tmux attach -t $SESSION"
