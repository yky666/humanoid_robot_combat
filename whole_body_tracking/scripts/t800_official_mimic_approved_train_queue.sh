#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_ROOT="${SRC_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
MANIFEST="${MANIFEST:-$REPO_ROOT/configs/t800_official_mimic_approved_20260902.json}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/tmux}"
PYTHON_BIN="${PYTHON_BIN:-}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-256}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
RUN_NAME="${RUN_NAME:-approved_official_mimic_5_t800_joint_v1_from29999}"
LOGGER="${LOGGER:-tensorboard}"
PROJECT="${PROJECT:-t800_urkl_qualifier_official_mimic}"
TASK_VARIANT="${TASK_VARIANT:-default}"
RESUME_LOAD_RUN="${RESUME_LOAD_RUN:-2026-08-31_16-18-44_motiondecode_t800_all_sys01_resume}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-model_29999.pt}"
PLAY_NUM_ENVS="${PLAY_NUM_ENVS:-25}"
PLAY_VIDEO_LENGTH="${PLAY_VIDEO_LENGTH:-1250}"
SWITCH_VIDEO_LENGTH="${SWITCH_VIDEO_LENGTH:-1250}"
SWITCH_INTERVAL="${SWITCH_INTERVAL:-250}"
SWITCH_MOTIONS="${SWITCH_MOTIONS:-}"
MIN_FREE_GPU_MEM_MB="${MIN_FREE_GPU_MEM_MB:-12000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-180}"
PROCESS_TIMEOUT="${PROCESS_TIMEOUT:-2400}"
TMPDIR_ROOT="${TMPDIR_ROOT:-$SRC_ROOT/tmp}"
CONVERT="${CONVERT:-1}"
TRAIN="${TRAIN:-1}"
PLAY="${PLAY:-1}"
PLAY_SWITCH="${PLAY_SWITCH:-1}"
TRAIN_OVERRIDES="${TRAIN_OVERRIDES:-agent.policy.noise_std_type=log agent.algorithm.learning_rate=5.0e-5 agent.algorithm.entropy_coef=0.002 agent.algorithm.desired_kl=0.005 agent.algorithm.max_grad_norm=0.5}"

mkdir -p "$LOG_DIR" "$TMPDIR_ROOT"
cd "$REPO_ROOT"

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in \
    /home/sys01/miniconda3/envs/env_isaaclab/bin/python \
    /data2/yangky/miniconda3/envs/env_isaaclab/bin/python \
    /data2/yangky/miniconda3/envs/gmr/bin/python; do
    if [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "[FAIL] Could not find a usable Python. Set PYTHON_BIN=/path/to/python."
  exit 1
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${REPO_ROOT}/source/whole_body_tracking:${SRC_ROOT}/IsaacLab/source/isaaclab:${SRC_ROOT}/IsaacLab/source/isaaclab_assets:${SRC_ROOT}/IsaacLab/source/isaaclab_contrib:${SRC_ROOT}/IsaacLab/source/isaaclab_mimic:${SRC_ROOT}/IsaacLab/source/isaaclab_rl:${SRC_ROOT}/IsaacLab/source/isaaclab_tasks:${PYTHONPATH:-}"
export TMPDIR="$TMPDIR_ROOT"
export OMNI_USER_CACHE_DIR="$TMPDIR_ROOT/omni_cache"
export OV_CACHE_ROOT="$TMPDIR_ROOT/ov_cache"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RESOLVED_MANIFEST="${RESOLVED_MANIFEST:-$LOG_DIR/$(basename "${MANIFEST%.json}")_$(hostname)_resolved.json}"

resolve_manifest() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$MANIFEST" "$RESOLVED_MANIFEST" <<'PY'
import json
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
manifest_path = pathlib.Path(sys.argv[2]).expanduser()
output_path = pathlib.Path(sys.argv[3]).expanduser()

with manifest_path.open(encoding="utf-8") as f:
    manifest = json.load(f)

def resolve(value):
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve())

for motion in manifest["motions"]:
    motion["input_file"] = resolve(motion["input_file"])
    motion["output_path"] = resolve(motion["output_path"])
    if "review_video" in motion:
        motion["review_video"] = resolve(motion["review_video"])

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(output_path)
PY
}

motion_rows() {
  "$PYTHON_BIN" - "$RESOLVED_MANIFEST" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
motions = json.loads(manifest.read_text(encoding="utf-8"))["motions"]
for motion in motions:
    print("\t".join([
        motion["name"],
        motion["input_file"],
        motion.get("input_format", "gmr_pickle"),
        motion["output_path"],
    ]))
PY
}

motion_names() {
  "$PYTHON_BIN" - "$RESOLVED_MANIFEST" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
motions = json.loads(manifest.read_text(encoding="utf-8"))["motions"]
print(" ".join(motion["name"] for motion in motions))
PY
}

validate_npz() {
  "$PYTHON_BIN" - "$1" <<'PY'
import numpy as np
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
required = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
if not path.is_file():
    raise SystemExit(f"missing: {path}")
with np.load(path) as data:
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit(f"missing keys in {path}: {missing}")
    if data["joint_pos"].ndim != 2 or data["body_pos_w"].ndim != 3:
        raise SystemExit(f"bad shapes in {path}")
    frames = int(data["joint_pos"].shape[0])
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
print(f"[OK] validated {path} frames={frames} fps={fps:g} duration={frames / fps:.2f}s")
PY
}

validate_manifest_npz() {
  "$PYTHON_BIN" - "$RESOLVED_MANIFEST" <<'PY'
import json
import numpy as np
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
bad = []
for motion in json.loads(manifest.read_text(encoding="utf-8"))["motions"]:
    path = pathlib.Path(motion["output_path"])
    if not path.is_file():
        bad.append(f"missing: {path}")
        continue
    try:
        with np.load(path) as data:
            frames = int(data["joint_pos"].shape[0])
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            print(f"[OK] {motion['name']}: frames={frames} fps={fps:g} duration={frames / fps:.2f}s")
    except Exception as exc:
        bad.append(f"{path}: {exc}")
if bad:
    print("\n".join(bad))
    raise SystemExit(1)
PY
}

stop_process_tree() {
  local pid="$1"
  if ! kill -0 "$pid" 2>/dev/null; then
    return
  fi
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! kill -0 "$pid" 2>/dev/null; then
      return
    fi
    sleep 1
  done
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
}

run_converter_until_npz() {
  local motion_name="$1"
  local input_file="$2"
  local input_format="$3"
  local output_path="$4"
  local output_name="$5"
  local start_time
  local converter_pid
  local validation
  local status

  start_time="$(date +%s)"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/t800_csv_to_npz.py" \
    --input_file "$input_file" \
    --input_format "$input_format" \
    --output_name "$output_name" \
    --output_path "$output_path" \
    --output_fps 50 \
    --skip_wandb_upload \
    --headless \
    --device "$DEVICE" &
  converter_pid=$!

  while true; do
    if validation="$(validate_npz "$output_path" 2>&1)"; then
      echo "$validation"
      if kill -0 "$converter_pid" 2>/dev/null; then
        echo "[INFO] $motion_name NPZ is valid; stopping converter pid=$converter_pid after save"
        stop_process_tree "$converter_pid"
      fi
      wait "$converter_pid" 2>/dev/null || true
      return 0
    fi

    if ! kill -0 "$converter_pid" 2>/dev/null; then
      set +e
      wait "$converter_pid"
      status=$?
      set -e
      if [[ "$status" -ne 0 ]]; then
        echo "[WARN] converter exited with status $status for $motion_name; checking whether a valid NPZ was written"
      fi
      validate_npz "$output_path"
      return 0
    fi

    if (( $(date +%s) - start_time >= PROCESS_TIMEOUT )); then
      echo "[FAIL] converter timed out for $motion_name after ${PROCESS_TIMEOUT}s"
      stop_process_tree "$converter_pid"
      return 1
    fi
    sleep 5
  done
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
echo "[INFO] repo: $REPO_ROOT"
echo "[INFO] source manifest: $MANIFEST"
echo "[INFO] python: $PYTHON_BIN"
resolve_manifest
echo "[INFO] resolved manifest: $RESOLVED_MANIFEST"

if [[ "$CONVERT" == "1" ]]; then
  while IFS=$'\t' read -r motion_name input_file input_format output_path; do
    if [[ ! -f "$input_file" ]]; then
      echo "[FAIL] missing input for $motion_name: $input_file"
      exit 1
    fi
    output_name="$(basename "${output_path%.npz}")"
    mkdir -p "$(dirname "$output_path")"
    if [[ -f "$output_path" ]]; then
      echo "[INFO] reusing existing NPZ for $motion_name: $output_path"
      validate_npz "$output_path"
      continue
    fi
    echo "[INFO] converting $motion_name -> $output_path"
    wait_for_gpu_memory
    run_converter_until_npz "$motion_name" "$input_file" "$input_format" "$output_path" "$output_name"
  done < <(motion_rows)
fi

validate_manifest_npz

if [[ "$TRAIN" != "1" ]]; then
  echo "[OK] conversion-only queue finished"
  exit 0
fi

train_overrides=()
if [[ -n "$TRAIN_OVERRIDES" ]]; then
  read -r -a train_overrides <<< "$TRAIN_OVERRIDES"
fi

echo "[INFO] training joint policy: $RUN_NAME"
echo "[INFO] max_iterations=$MAX_ITERATIONS num_envs=$NUM_ENVS resume=$RESUME_LOAD_RUN/$RESUME_CHECKPOINT"
wait_for_gpu_memory
train_log="$LOG_DIR/train_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/rsl_rl/train_t800.py" \
  --motion_file "$RESOLVED_MANIFEST" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --device "$DEVICE" \
  --run_name "$RUN_NAME" \
  --logger "$LOGGER" \
  --log_project_name "$PROJECT" \
  --task_variant "$TASK_VARIANT" \
  --headless \
  -- \
  --resume true \
  --load_run "$RESUME_LOAD_RUN" \
  --checkpoint "$RESUME_CHECKPOINT" \
  "${train_overrides[@]}" 2>&1 | tee -a "$train_log"
train_status=${PIPESTATUS[0]}
set -e
if [[ "$train_status" -ne 0 ]]; then
  echo "[FAIL] training exited with status $train_status"
  exit "$train_status"
fi

run_dir="$(latest_run_dir)"
if [[ -z "$run_dir" ]]; then
  echo "[FAIL] could not find run directory for $RUN_NAME"
  exit 1
fi
checkpoint="$(latest_checkpoint "$run_dir")"
if [[ -z "$checkpoint" ]]; then
  echo "[FAIL] could not find checkpoint in $run_dir"
  exit 1
fi

if [[ "$PLAY" == "1" ]]; then
  echo "[INFO] rendering trained policy: $run_dir/$checkpoint"
  play_log="$LOG_DIR/play_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
  wait_for_gpu_memory
  set +e
  "$PYTHON_BIN" "$REPO_ROOT/scripts/rsl_rl/play_t800.py" \
    --load_run "$run_dir" \
    --checkpoint "$checkpoint" \
    --motion_file "$RESOLVED_MANIFEST" \
    --num_envs "$PLAY_NUM_ENVS" \
    --device "$DEVICE" \
    --task_variant "$TASK_VARIANT" \
    --video \
    --video_length "$PLAY_VIDEO_LENGTH" \
    --headless 2>&1 | tee -a "$play_log"
  play_status=${PIPESTATUS[0]}
  set -e
  if [[ "$play_status" -ne 0 ]]; then
    echo "[WARN] playback render exited with status $play_status"
  fi
fi

if [[ "$PLAY_SWITCH" == "1" && -f "$REPO_ROOT/scripts/rsl_rl/play_t800_policy_switch.py" ]]; then
  echo "[INFO] rendering explicit mode-switch playback"
  switch_log="$LOG_DIR/play_switch_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
  if [[ -z "$SWITCH_MOTIONS" ]]; then
    SWITCH_MOTIONS="$(motion_names)"
  fi
  read -r -a switch_motions <<< "$SWITCH_MOTIONS"
  wait_for_gpu_memory
  set +e
  "$PYTHON_BIN" "$REPO_ROOT/scripts/rsl_rl/play_t800_policy_switch.py" \
    --load_run "$run_dir" \
    --checkpoint "$checkpoint" \
    --motion_file "$RESOLVED_MANIFEST" \
    --switch_motions "${switch_motions[@]}" \
    --switch_interval "$SWITCH_INTERVAL" \
    --video \
    --video_length "$SWITCH_VIDEO_LENGTH" \
    --num_envs 5 \
    --device "$DEVICE" \
    --task_variant "$TASK_VARIANT" \
    --video_folder_name approved_mimic_switch \
    --headless 2>&1 | tee -a "$switch_log"
  switch_status=${PIPESTATUS[0]}
  set -e
  if [[ "$switch_status" -ne 0 ]]; then
    echo "[WARN] mode-switch render exited with status $switch_status"
  fi
fi

echo "[OK] queue finished"
echo "[OK] checkpoint: $run_dir/$checkpoint"
echo "[OK] exported: $run_dir/exported"
echo "[OK] videos: $run_dir/videos"
