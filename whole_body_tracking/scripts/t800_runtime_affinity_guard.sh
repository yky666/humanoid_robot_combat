#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CPUSET="${T800_CPUSET:-0-$(( $(nproc) - 1 ))}"
INTERVAL="${T800_AFFINITY_INTERVAL:-5}"
STATE_ROOT="${STATE_ROOT:-$REPO_ROOT/artifacts/t800_qualifier_training_canonical_v1_20260902}"
LOG_FILE="${T800_AFFINITY_LOG:-$STATE_ROOT/logs/affinity_guard_$(date +%Y%m%d_%H%M%S).log}"
PROCESS_PATTERN="$REPO_ROOT/scripts/rsl_rl/(train|evaluate_t800_policy|play_t800|play_t800_policy_switch)\.py"
QUEUE_PATTERN="$REPO_ROOT/scripts/t800_qualifier_staged_train_queue\.sh"

mkdir -p "$(dirname "$LOG_FILE")"
printf '%s start cpuset=%s interval=%s\n' "$(date --iso-8601=seconds)" "$CPUSET" "$INTERVAL" >> "$LOG_FILE"

while pgrep -f "$QUEUE_PATTERN" >/dev/null; do
  mapfile -t pids < <(pgrep -f "$PROCESS_PATTERN" || true)
  for pid in "${pids[@]}"; do
    [[ -d "/proc/$pid" ]] || continue
    before="$(taskset -pc "$pid" 2>/dev/null | awk -F: '{gsub(/^ /, "", $2); print $2}')"
    if [[ -n "$before" && "$before" != "$CPUSET" ]]; then
      taskset -pc "$CPUSET" "$pid" >/dev/null 2>&1 || continue
      printf '%s widened pid=%s before=%s after=%s\n' \
        "$(date --iso-8601=seconds)" "$pid" "$before" "$CPUSET" >> "$LOG_FILE"
    fi
  done
  sleep "$INTERVAL"
done

printf '%s stop queue_not_running\n' "$(date --iso-8601=seconds)" >> "$LOG_FILE"
