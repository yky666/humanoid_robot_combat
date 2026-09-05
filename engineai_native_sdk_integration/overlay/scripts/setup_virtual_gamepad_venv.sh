#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${1:-$ROOT_DIR/.venv_gamepad}"

python3 -m venv --system-site-packages "$VENV_DIR"
"$VENV_DIR/bin/python3" -m pip install --upgrade pip

if "$VENV_DIR/bin/python3" - <<'PY'
import PyQt6
PY
then
  echo "[INFO] PyQt6 is available from system site-packages."
else
  REQ_NO_LCM="$(mktemp)"
  grep -vE '^lcm([<>= ]|$)' "$ROOT_DIR/tools/virtual_gamepad/requirements.txt" > "$REQ_NO_LCM"
  "$VENV_DIR/bin/python3" -m pip install -r "$REQ_NO_LCM"
  rm -f "$REQ_NO_LCM"
fi

"$VENV_DIR/bin/python3" - <<'PY'
import lcm
import PyQt6
PY

echo "[INFO] Virtual gamepad venv ready: $VENV_DIR"
echo "[INFO] Test combos with:"
echo "       $ROOT_DIR/scripts/send_t800_qualifier_key.sh --list"
