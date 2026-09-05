#!/bin/bash

set -u

echo "timestamp: $(date --iso-8601=seconds)"
echo "robotics.service: $(systemctl is-active robotics.service 2>/dev/null || true)"

mapfile -t pids < <(pgrep -f '^.*/_install/bin/src_executor t800$' || true)
if ((${#pids[@]} == 0)); then
  echo "custom executor: stopped"
  echo "fsm state: unavailable (no controller process)"
else
  echo "custom executor count: ${#pids[@]}"
  for pid in "${pids[@]}"; do
    root=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
    exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
    echo "pid $pid"
    echo "  cwd: ${root:-unavailable}"
    echo "  exe: ${exe:-unavailable}"

    if [[ -n $root && -d $root/runtime_logs ]]; then
      log=$(find "$root/runtime_logs" -maxdepth 1 -name 'custom_*.log' \
        -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
      if [[ -n $log ]]; then
        state=$(grep 'Entered motion' "$log" 2>/dev/null | tail -n 1 || true)
        echo "  log: $log"
        echo "  fsm: ${state:-no entered-motion record}"
      fi
    fi
  done
fi

if command -v ethercat >/dev/null 2>&1; then
  echo "ethercat slaves:"
  ethercat slaves 2>&1 || true
else
  echo "ethercat slaves: unavailable (command not found)"
fi
