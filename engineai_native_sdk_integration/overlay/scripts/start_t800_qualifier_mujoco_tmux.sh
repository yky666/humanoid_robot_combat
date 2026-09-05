#!/bin/bash

set -euo pipefail

SESSION_NAME="${1:-t800_qualifier_mujoco}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODE_DIR="$ROOT_DIR/assets/config/t800"
DEFAULT_MODE="$MODE_DIR/mode.yaml"
QUALIFIER_MODE="${QUALIFIER_MODE:-$MODE_DIR/mode_qualifier_approved.yaml}"
BACKUP_MODE="$MODE_DIR/mode.yaml.before_qualifier_tmux.$(date +%Y%m%d_%H%M%S)"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[INFO] tmux session already exists: $SESSION_NAME"
  echo "       tmux attach -t $SESSION_NAME"
  exit 0
fi

if [ ! -f "$QUALIFIER_MODE" ]; then
  echo "[ERROR] Missing qualifier mode config: $QUALIFIER_MODE" >&2
  exit 1
fi

if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ] && [ "${ALLOW_HEADLESS:-0}" != "1" ]; then
  echo "[ERROR] DISPLAY/WAYLAND_DISPLAY is empty; MuJoCo GUI needs a graphical session or X11 forwarding." >&2
  echo "        Re-run from a graphical terminal, or set ALLOW_HEADLESS=1 for a non-visible smoke test." >&2
  exit 1
fi

cp "$DEFAULT_MODE" "$BACKUP_MODE"
cp "$QUALIFIER_MODE" "$DEFAULT_MODE"

common_env="export LD_LIBRARY_PATH=/opt/engineai_robotics_third_party/lib:/opt/engineai_robotics_hardware/lib:$ROOT_DIR/core/lib:$ROOT_DIR/build/_install/lib:/opt/onnxruntime/lib:\${LD_LIBRARY_PATH:-}; export ENGINEAI_ROBOTICS_THIRD_PARTY=/opt/engineai_robotics_third_party; export ENGINEAI_ROBOTICS_HARDWARE=/opt/engineai_robotics_hardware"
gamepad_python="$ROOT_DIR/.venv_gamepad/bin/python3"
if [ ! -x "$gamepad_python" ]; then
  gamepad_python="python3"
fi

tmux new-session -d -s "$SESSION_NAME" -n mujoco \
  "cd '$ROOT_DIR'; $common_env; ./scripts/run_mujoco.sh t800; exec bash"

tmux new-window -t "$SESSION_NAME" -n executor \
  "cd '$ROOT_DIR'; $common_env; sleep 2; ./run.sh t800; exec bash"

tmux new-window -t "$SESSION_NAME" -n gamepad \
  "cd '$ROOT_DIR'; $common_env; if [ -z \"\${DISPLAY:-}\" ] && [ -z \"\${WAYLAND_DISPLAY:-}\" ]; then echo '[WARN] DISPLAY/WAYLAND_DISPLAY is empty; use ./scripts/send_t800_qualifier_key.sh <motion> from a terminal.'; elif '$gamepad_python' -c 'import lcm, PyQt6' >/dev/null 2>&1; then '$gamepad_python' tools/virtual_gamepad/virtual_gamepad.py; else echo '[WARN] Missing Python lcm/PyQt6; run ./scripts/setup_virtual_gamepad_venv.sh or use ./scripts/send_t800_qualifier_key.sh <motion>.'; fi; exec bash"

cat <<EOF
[INFO] Started tmux session: $SESSION_NAME
[INFO] Attach with:
       tmux attach -t $SESSION_NAME
[INFO] Activated qualifier mode:
       $DEFAULT_MODE
[INFO] Backup:
       $BACKUP_MODE

Key map:
  LB+A       pd_stand
  LB+B       walk
  RB+B       native dance
  RB+A       qualifier_front_kick
  RB+X       qualifier_spinning_kick
  RB+Y       qualifier_straight_punch
  LB+X       qualifier_hook_punch
  LB+Y       qualifier_jab_left
  BACK+A     qualifier_recovery_supine
  LB+RB      passive

Keyboard equivalents in the virtual gamepad window:
  LB=q, RB=e, A=j, B=k, X=u, Y=i, BACK=F1, START=F2
  W/S=forward/backward, A/D=left/right (walk mode; release to stop)
  Shift+Left/Right=yaw, Space=reset all stick axes

Keyboard-only CLI sender:
  ./scripts/send_t800_qualifier_key.sh pd_stand
  ./scripts/send_t800_qualifier_key.sh front_kick
  ./scripts/send_t800_qualifier_key.sh spinning_kick
  ./scripts/send_t800_qualifier_key.sh straight_punch
  ./scripts/send_t800_qualifier_key.sh hook_punch
  ./scripts/send_t800_qualifier_key.sh jab_left
  ./scripts/send_t800_qualifier_key.sh recovery_supine

Optional virtual gamepad setup:
  ./scripts/setup_virtual_gamepad_venv.sh
EOF
