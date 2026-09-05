#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_ROOT="${SRC_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$SRC_ROOT/datasets/urkl_locomotion_260901}"
PKL_ROOT="${PKL_ROOT:-$DATA_ROOT/gmr_pkl_all}"
NPZ_ROOT="${NPZ_ROOT:-$DATA_ROOT/tracking_npz}"
INPUT_SUFFIX="${INPUT_SUFFIX:-pkl}"
INPUT_FORMAT="${INPUT_FORMAT:-gmr_pickle}"
GMR_SESSION="${GMR_SESSION:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
PROCESS_TIMEOUT="${PROCESS_TIMEOUT:-1200}"
FORCE="${FORCE:-0}"

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in \
    /data2/yangky/miniconda3/envs/env_isaaclab/bin/python \
    /home/sys01/miniconda3/envs/env_isaaclab/bin/python \
    /data2/yangky/miniconda3/envs/isaaclab/bin/python; do
    if [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "[FAIL] Could not find IsaacLab Python. Set PYTHON_BIN=/path/to/python."
  exit 1
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${REPO_ROOT}/source/whole_body_tracking:${SRC_ROOT}/IsaacLab/source/isaaclab:${SRC_ROOT}/IsaacLab/source/isaaclab_assets:${SRC_ROOT}/IsaacLab/source/isaaclab_contrib:${SRC_ROOT}/IsaacLab/source/isaaclab_mimic:${SRC_ROOT}/IsaacLab/source/isaaclab_rl:${SRC_ROOT}/IsaacLab/source/isaaclab_tasks:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-$SRC_ROOT/tmp}"
export OMNI_USER_CACHE_DIR="${OMNI_USER_CACHE_DIR:-$TMPDIR/omni_cache}"
export OV_CACHE_ROOT="${OV_CACHE_ROOT:-$TMPDIR/ov_cache}"

validate_npz() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
import numpy as np

required = {
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}
try:
    with np.load(sys.argv[1]) as data:
        ok = required.issubset(data.files) and data["joint_pos"].ndim == 2 and data["body_pos_w"].ndim == 3
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

if [[ -n "$GMR_SESSION" ]]; then
  echo "[INFO] waiting for GMR tmux session: $GMR_SESSION"
  while tmux has-session -t "$GMR_SESSION" 2>/dev/null; do
    sleep 60
  done
fi

if [[ ! -d "$PKL_ROOT" ]]; then
  echo "[FAIL] Missing PKL_ROOT: $PKL_ROOT"
  exit 1
fi

mkdir -p "$NPZ_ROOT"
mapfile -d '' pkl_files < <(find "$PKL_ROOT" -type f -name "*.${INPUT_SUFFIX}" -print0 | sort -z)
if (( ${#pkl_files[@]} == 0 )); then
  echo "[FAIL] No *.${INPUT_SUFFIX} files found under $PKL_ROOT"
  exit 1
fi

echo "[INFO] converting ${#pkl_files[@]} *.${INPUT_SUFFIX} files to tracking npz"
for pkl_file in "${pkl_files[@]}"; do
  rel_path="${pkl_file#$PKL_ROOT/}"
  output_path="$NPZ_ROOT/${rel_path%.*}_tracking.npz"
  output_name="$(basename "${output_path%.npz}")"
  mkdir -p "$(dirname "$output_path")"

  if [[ "$FORCE" != "1" && -f "$output_path" ]] && validate_npz "$output_path"; then
    echo "[SKIP] valid npz exists: $output_path"
    continue
  fi

  echo "[RUN ] $pkl_file"
  set +e
  timeout "$PROCESS_TIMEOUT" "$PYTHON_BIN" "$REPO_ROOT/scripts/t800_csv_to_npz.py" \
    --input_file "$pkl_file" \
    --input_format "$INPUT_FORMAT" \
    --output_name "$output_name" \
    --output_path "$output_path" \
    --output_fps "$OUTPUT_FPS" \
    --skip_wandb_upload \
    --headless \
    --device "$DEVICE"
  status=$?
  set -e
  if (( status != 0 )); then
    if [[ -f "$output_path" ]] && validate_npz "$output_path"; then
      echo "[WARN] converter exited with status $status after writing a valid npz; continuing"
    else
      echo "[FAIL] converter exited with status $status and did not produce a valid npz: $output_path"
      exit "$status"
    fi
  fi
done

echo "[OK] locomotion npz queue finished"
