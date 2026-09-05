#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODE_DIR="$ROOT_DIR/assets/config/t800"
DEFAULT_MODE="$MODE_DIR/mode.yaml"
QUALIFIER_MODE="$MODE_DIR/mode_qualifier_approved.yaml"
BACKUP_MODE="$MODE_DIR/mode.yaml.before_qualifier.$(date +%Y%m%d_%H%M%S)"

if [ ! -f "$QUALIFIER_MODE" ]; then
    echo "[ERROR] Missing qualifier mode config: $QUALIFIER_MODE" >&2
    exit 1
fi

cp "$DEFAULT_MODE" "$BACKUP_MODE"
restore_mode() {
    cp "$BACKUP_MODE" "$DEFAULT_MODE"
    echo "[INFO] Restored mode.yaml from $BACKUP_MODE"
}
trap restore_mode EXIT

cp "$QUALIFIER_MODE" "$DEFAULT_MODE"
echo "[INFO] Activated T800 qualifier mode for this MuJoCo session."
echo "[INFO] Start virtual gamepad separately if needed:"
echo "       python3 $ROOT_DIR/tools/virtual_gamepad/virtual_gamepad.py"
echo "[INFO] Keyboard-only mode switching:"
echo "       $ROOT_DIR/scripts/send_t800_qualifier_key.sh front_kick"

"$ROOT_DIR/scripts/run_mujoco.sh" t800
