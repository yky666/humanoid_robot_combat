#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SESSION="${SESSION:-t800_qualifier_staged_train}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/artifacts/t800_qualifier_training_canonical_v1_20260902/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/queue_$(date +%Y%m%d_%H%M%S).log}"

quote() {
  printf "%q" "$1"
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[FAIL] tmux session already exists: $SESSION"
  echo "[INFO] attach: tmux attach -t $SESSION"
  exit 1
fi

mkdir -p "$LOG_DIR"
command="$(quote "$REPO_ROOT/scripts/t800_qualifier_staged_train_queue.sh")"
for argument in "$@"; do
  command+=" $(quote "$argument")"
done
command+=" 2>&1 | tee -a $(quote "$LOG_FILE")"
command+="; status=\${PIPESTATUS[0]}"
command+="; echo [INFO] staged queue exited with status \$status"
command+="; echo [INFO] log: $(quote "$LOG_FILE")"
command+="; exec bash"

tmux new-session -d -s "$SESSION" "bash -lc $(quote "$command")"

echo "[OK] tmux session: $SESSION"
echo "[OK] log: $LOG_FILE"
echo "[OK] attach: tmux attach -t $SESSION"
