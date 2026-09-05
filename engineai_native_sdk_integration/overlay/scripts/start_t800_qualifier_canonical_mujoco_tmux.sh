#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ASSET_TAG="rl_qualifier_canonical_v1_20260902"
READY_FILE="$ROOT_DIR/assets/config/t800/$ASSET_TAG/READY.json"
QUALIFIER_MODE="$ROOT_DIR/assets/config/t800/mode_qualifier_canonical_v1.yaml"

if [[ ! -f "$READY_FILE" || ! -f "$QUALIFIER_MODE" ]]; then
  echo "[ERROR] Canonical qualifier policies have not passed and been staged yet." >&2
  echo "        Missing: $READY_FILE or $QUALIFIER_MODE" >&2
  exit 1
fi

export QUALIFIER_MODE
exec "$SCRIPT_DIR/start_t800_qualifier_mujoco_tmux.sh" "${1:-t800_qualifier_canonical_mujoco}"
