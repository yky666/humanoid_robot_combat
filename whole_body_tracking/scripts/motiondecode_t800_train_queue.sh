#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data2/yangky/test/whole_body_tracking}"
MANIFEST="${MANIFEST:-/data2/yangky/test/datasets/MotionDecode_T800_aligned/tracking_manifest_csv.json}"
LOG_DIR="${LOG_DIR:-/data2/yangky/test/datasets/MotionDecode_T800_aligned/logs}"
PYTHON_BIN="${PYTHON_BIN:-/data2/yangky/miniconda3/envs/env_isaaclab/bin/python}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-128}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
RUN_NAME="${RUN_NAME:-motiondecode_t800_all}"
LOGGER="${LOGGER:-tensorboard}"
PROJECT="${PROJECT:-t800_motiondecode}"
TASK_VARIANT="${TASK_VARIANT:-default}"
POLL_SECONDS="${POLL_SECONDS:-300}"
PLAY_NUM_ENVS="${PLAY_NUM_ENVS:-25}"
PLAY_VIDEO_LENGTH="${PLAY_VIDEO_LENGTH:-1000}"
TMPDIR_ROOT="${TMPDIR_ROOT:-/data2/yangky/test/tmp}"
MIN_FREE_GPU_MEM_MB="${MIN_FREE_GPU_MEM_MB:-9000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-300}"

mkdir -p "$LOG_DIR" "$TMPDIR_ROOT"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
export TMPDIR="$TMPDIR_ROOT"
export OMNI_USER_CACHE_DIR="$TMPDIR_ROOT/omni_cache"
export OV_CACHE_ROOT="$TMPDIR_ROOT/ov_cache"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

status_line() {
  "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
motions = json.loads(manifest.read_text())["motions"]
paths = [pathlib.Path(m["output_path"]) for m in motions]
existing = sum(path.is_file() for path in paths)
print(existing, len(paths), len(paths) - existing)
PY
}

validate_all_npz() {
  "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import numpy as np
import pathlib
import sys

required = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
manifest = pathlib.Path(sys.argv[1])
motions = json.loads(manifest.read_text())["motions"]
bad = []
for motion in motions:
    path = pathlib.Path(motion["output_path"])
    if not path.is_file():
        bad.append(f"missing: {path}")
        continue
    try:
        with np.load(path) as data:
            missing_keys = [key for key in required if key not in data]
            if missing_keys:
                bad.append(f"bad_keys: {path}: {missing_keys}")
                continue
            if data["joint_pos"].ndim != 2 or data["body_pos_w"].ndim != 3:
                bad.append(f"bad_shape: {path}")
    except Exception as exc:
        bad.append(f"load_failed: {path}: {exc}")
if bad:
    print("\n".join(bad[:20]))
    if len(bad) > 20:
        print(f"... {len(bad) - 20} more")
    raise SystemExit(1)
print(f"[OK] validated {len(motions)} NPZ files")
PY
}

wait_for_gpu_memory() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi not found; skipping GPU memory wait"
    return
  fi

  while true; do
    local free_mem
    free_mem="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    echo "[INFO] GPU${GPU} free memory: ${free_mem}MiB, required: ${MIN_FREE_GPU_MEM_MB}MiB"
    if [[ "$free_mem" =~ ^[0-9]+$ ]] && (( free_mem >= MIN_FREE_GPU_MEM_MB )); then
      break
    fi
    sleep "$GPU_POLL_SECONDS"
  done
}

latest_run_dir() {
  find "$REPO_ROOT/logs/rsl_rl/t800_flat" -maxdepth 1 -type d -name "*_${RUN_NAME}" -printf "%T@ %p\n" 2>/dev/null \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

latest_checkpoint() {
  local run_dir="$1"
  find "$run_dir" -maxdepth 1 -type f -name "model_*.pt" -printf "%f\n" \
    | sort -V \
    | tail -n 1
}

echo "[INFO] queue started at $(date --iso-8601=seconds)"
echo "[INFO] manifest: $MANIFEST"
echo "[INFO] waiting for all converted NPZ files..."

while true; do
  read -r existing total missing < <(status_line)
  echo "[INFO] NPZ progress: ${existing}/${total}, missing=${missing} at $(date --iso-8601=seconds)"
  if [[ "$missing" == "0" ]]; then
    validate_all_npz
    break
  fi
  sleep "$POLL_SECONDS"
done

wait_for_gpu_memory
train_log="$LOG_DIR/train_motiondecode_t800_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] starting training, log: $train_log"

set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/rsl_rl/train_t800.py" \
  --motion_file "$MANIFEST" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --device "$DEVICE" \
  --run_name "$RUN_NAME" \
  --logger "$LOGGER" \
  --log_project_name "$PROJECT" \
  --task_variant "$TASK_VARIANT" \
  --headless \
  "$@" 2>&1 | tee -a "$train_log"
train_status=${PIPESTATUS[0]}
set -e
if [[ "$train_status" -ne 0 ]]; then
  echo "[FAIL] training exited with status $train_status"
  exit "$train_status"
fi

run_dir="$(latest_run_dir)"
if [[ -z "$run_dir" ]]; then
  echo "[FAIL] could not find training run directory for run name: $RUN_NAME"
  exit 1
fi

checkpoint="$(latest_checkpoint "$run_dir")"
if [[ -z "$checkpoint" ]]; then
  echo "[FAIL] could not find model_*.pt in run directory: $run_dir"
  exit 1
fi

play_log="$LOG_DIR/play_motiondecode_t800_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] training complete: $run_dir/$checkpoint"
echo "[INFO] starting playback export/render, log: $play_log"

wait_for_gpu_memory
set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/rsl_rl/play_t800.py" \
  --load_run "$run_dir" \
  --checkpoint "$checkpoint" \
  --motion_file "$MANIFEST" \
  --num_envs "$PLAY_NUM_ENVS" \
  --device "$DEVICE" \
  --task_variant "$TASK_VARIANT" \
  --video \
  --video_length "$PLAY_VIDEO_LENGTH" \
  --headless 2>&1 | tee -a "$play_log"
play_status=${PIPESTATUS[0]}
set -e
if [[ "$play_status" -ne 0 ]]; then
  echo "[FAIL] playback/export exited with status $play_status"
  exit "$play_status"
fi

echo "[OK] queue finished"
echo "[OK] checkpoint: $run_dir/$checkpoint"
echo "[OK] onnx: $run_dir/exported/policy.onnx"
echo "[OK] videos: $run_dir/videos/play"
