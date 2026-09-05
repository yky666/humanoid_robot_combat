#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-motiondecode_t800_train_queue}"
REPO_ROOT="${REPO_ROOT:-/data2/yangky/test/whole_body_tracking}"
LOG_DIR="${LOG_DIR:-/data2/yangky/test/datasets/MotionDecode_T800_aligned/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_queue_motiondecode_t800_$(date +%Y%m%d_%H%M%S).log}"

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

mkdir -p "$LOG_DIR"

cmd="set -euo pipefail"
cmd+="; set +e"
cmd+="; $(quote "$REPO_ROOT/scripts/motiondecode_t800_train_queue.sh")$(join_quoted "$@") 2>&1 | tee -a $(quote "$LOG_FILE")"
cmd+="; status=\${PIPESTATUS[0]}"
cmd+="; echo [INFO] train queue exited with status \$status"
cmd+="; echo [INFO] log: $(quote "$LOG_FILE")"
cmd+="; echo [INFO] press Ctrl-b d to detach, or exit to close this shell"
cmd+="; exit \$status"

tmux new-session -d -s "$SESSION" "bash -lc $(quote "$cmd")"

echo "[OK] started tmux session: $SESSION"
echo "[OK] log: $LOG_FILE"
echo "[OK] attach: tmux attach -t $SESSION"
