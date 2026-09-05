#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_ROOT="${SRC_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
MANIFEST="${MANIFEST:-$REPO_ROOT/configs/t800_combat_motion_manifest.json}"
QC_MANIFEST="${QC_MANIFEST:-$REPO_ROOT/configs/t800_combat_reference_qc.json}"
REQUIRE_QC_PASS="${REQUIRE_QC_PASS:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"
GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
LOGGER="${LOGGER:-tensorboard}"
PROJECT="${PROJECT:-t800_urkl_qualifier_missing}"
TASK_VARIANT="${TASK_VARIANT:-default}"
RESUME_LOAD_RUN="${RESUME_LOAD_RUN:-2026-08-31_16-18-44_motiondecode_t800_all_sys01_resume}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-model_29999.pt}"
PLAY_NUM_ENVS="${PLAY_NUM_ENVS:-25}"
PLAY_VIDEO_LENGTH="${PLAY_VIDEO_LENGTH:-1000}"
MIN_FREE_GPU_MEM_MB="${MIN_FREE_GPU_MEM_MB:-18000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-300}"
TMPDIR_ROOT="${TMPDIR_ROOT:-$REPO_ROOT/../tmp}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/tmux}"
MOTIONS="${MOTIONS:-accad_e1_jab_left accad_e2_jab_right accad_e5_hook_left accad_e6_hook_right}"
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

motion_info() {
  "$PYTHON_BIN" - "$MANIFEST" "$1" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
motion_name = sys.argv[2]
motions = json.loads(manifest.read_text())["motions"]
for motion in motions:
    if motion["name"] == motion_name:
        print(motion["output_path"])
        print(motion["formal_run_name"])
        raise SystemExit(0)
raise SystemExit(f"motion not found in manifest: {motion_name}")
PY
}

motion_qc_status() {
  "$PYTHON_BIN" - "$QC_MANIFEST" "$1" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
motion_name = sys.argv[2]
if not manifest.is_file():
    print("missing_qc_manifest")
    raise SystemExit(0)
motions = json.loads(manifest.read_text()).get("motions", {})
entry = motions.get(motion_name)
if entry is None:
    print("missing_qc_entry")
elif isinstance(entry, str):
    print(entry)
else:
    print(entry.get("status", "missing_qc_status"))
PY
}

latest_run_dir() {
  local run_name="$1"
  find "$REPO_ROOT/logs/rsl_rl/t800_flat" -maxdepth 1 -type d -name "*_${run_name}" -printf "%T@ %p\n" 2>/dev/null \
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
echo "[INFO] manifest: $MANIFEST"
echo "[INFO] qc manifest: $QC_MANIFEST"
echo "[INFO] require qc pass: $REQUIRE_QC_PASS"
echo "[INFO] motions: $MOTIONS"
echo "[INFO] train overrides: $TRAIN_OVERRIDES"
echo "[INFO] python: $PYTHON_BIN"

for motion_name in $MOTIONS; do
  train_overrides=()
  if [[ -n "$TRAIN_OVERRIDES" ]]; then
    read -r -a train_overrides <<< "$TRAIN_OVERRIDES"
  fi
  mapfile -t info < <(motion_info "$motion_name")
  motion_file="${info[0]}"
  run_name="${info[1]}"
  if [[ ! -f "$motion_file" ]]; then
    echo "[FAIL] Missing motion file for $motion_name: $motion_file"
    exit 1
  fi
  if [[ "$REQUIRE_QC_PASS" == "1" ]]; then
    qc_status="$(motion_qc_status "$motion_name")"
    echo "[INFO] reference qc for $motion_name: $qc_status"
    if [[ "$qc_status" != "pass" ]]; then
      echo "[FAIL] Refusing to train $motion_name because reference QC is not pass. Set REQUIRE_QC_PASS=0 to override."
      exit 1
    fi
  fi

  echo "[INFO] training $motion_name -> $run_name"
  wait_for_gpu_memory
  "$PYTHON_BIN" "$REPO_ROOT/scripts/rsl_rl/train_t800.py" \
    --motion_file "$motion_file" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --device "$DEVICE" \
    --run_name "$run_name" \
    --logger "$LOGGER" \
    --log_project_name "$PROJECT" \
    --task_variant "$TASK_VARIANT" \
    --headless \
    -- \
    --resume true \
    --load_run "$RESUME_LOAD_RUN" \
    --checkpoint "$RESUME_CHECKPOINT" \
    "${train_overrides[@]}"

  run_dir="$(latest_run_dir "$run_name")"
  if [[ -z "$run_dir" ]]; then
    echo "[FAIL] Could not find run directory for $run_name"
    exit 1
  fi
  checkpoint="$(latest_checkpoint "$run_dir")"
  if [[ -z "$checkpoint" ]]; then
    echo "[FAIL] Could not find checkpoint in $run_dir"
    exit 1
  fi

  echo "[INFO] rendering $motion_name from $run_dir/$checkpoint"
  wait_for_gpu_memory
  "$PYTHON_BIN" "$REPO_ROOT/scripts/rsl_rl/play_t800.py" \
    --load_run "$run_dir" \
    --checkpoint "$checkpoint" \
    --motion_file "$motion_file" \
    --num_envs "$PLAY_NUM_ENVS" \
    --device "$DEVICE" \
    --task_variant "$TASK_VARIANT" \
    --video \
    --video_length "$PLAY_VIDEO_LENGTH" \
    --headless

  echo "[OK] finished $motion_name"
done

echo "[OK] queue finished at $(date --iso-8601=seconds)"
