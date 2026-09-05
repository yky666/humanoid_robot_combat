#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

export LD_LIBRARY_PATH="/opt/engineai_robotics_third_party/lib:/opt/engineai_robotics_hardware/lib:$ROOT_DIR/core/lib:$ROOT_DIR/build/_install/lib:/opt/onnxruntime/lib:${LD_LIBRARY_PATH:-}"
export ENGINEAI_ROBOTICS_THIRD_PARTY="${ENGINEAI_ROBOTICS_THIRD_PARTY:-/opt/engineai_robotics_third_party}"
export ENGINEAI_ROBOTICS_HARDWARE="${ENGINEAI_ROBOTICS_HARDWARE:-/opt/engineai_robotics_hardware}"

GAMEPAD_PYTHON="$ROOT_DIR/.venv_gamepad/bin/python3"
if [ ! -x "$GAMEPAD_PYTHON" ]; then
  GAMEPAD_PYTHON="python3"
fi

"$GAMEPAD_PYTHON" "$ROOT_DIR/tools/virtual_gamepad/send_gamepad_combo.py" "$@"
