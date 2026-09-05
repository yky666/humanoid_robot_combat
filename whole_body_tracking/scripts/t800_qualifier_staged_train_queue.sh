#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_ROOT="${SRC_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$REPO_ROOT/configs/t800_qualifier_canonical_v1_20260902.json}"
STATE_ROOT="${STATE_ROOT:-$REPO_ROOT/artifacts/t800_qualifier_training_canonical_v1_20260902}"
PYTHON_BIN="${PYTHON_BIN:-/home/sys01/miniconda3/envs/env_isaaclab/bin/python}"
GMR_PYTHON_BIN="${GMR_PYTHON_BIN:-}"
GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-512}"
EVAL_ENVS="${EVAL_ENVS:-64}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
ITERATIONS_PER_ROUND="${ITERATIONS_PER_ROUND:-5000}"
SINGLE_MAX_ROUNDS="${SINGLE_MAX_ROUNDS:-6}"
RECOVERY_MAX_ROUNDS="${RECOVERY_MAX_ROUNDS:-8}"
JOINT_MAX_ROUNDS="${JOINT_MAX_ROUNDS:-8}"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-0.95}"
MIN_FREE_GPU_MEM_MB="${MIN_FREE_GPU_MEM_MB:-12000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-120}"
TASK_VARIANT="${TASK_VARIANT:-default}"
LOGGER="${LOGGER:-tensorboard}"
PROJECT="${PROJECT:-t800_qualifier_canonical_v1}"
PLAY_NUM_ENVS="${PLAY_NUM_ENVS:-9}"
PLAY_VIDEO_LENGTH="${PLAY_VIDEO_LENGTH:-500}"
TRAIN_OVERRIDES="${TRAIN_OVERRIDES:-agent.policy.noise_std_type=log agent.algorithm.learning_rate=2.0e-4 agent.algorithm.entropy_coef=0.005 agent.algorithm.desired_kl=0.01 agent.algorithm.max_grad_norm=0.5}"
STANDING_MOTION_ORDER="${STANDING_MOTION_ORDER:-front_kick,spinning_kick,straight_punch,hook_punch,jab_left}"

LOG_DIR="$STATE_ROOT/logs"
REPORT_DIR="$STATE_ROOT/reports"
BEST_DIR="$STATE_ROOT/best"
TRANSITION_RAW_DIR="$STATE_ROOT/stand_action_stand/gmr_npz"
TRANSITION_TRACKING_DIR="$STATE_ROOT/stand_action_stand/tracking_npz"
TRANSITION_MANIFEST="$STATE_ROOT/stand_action_stand/manifest.json"
TRANSITION_REVIEW_DIR="$REPO_ROOT/review_videos/t800_stand_action_stand_canonical_v1_20260902"
RESOLVED_MANIFEST="$STATE_ROOT/source_manifest_resolved.json"

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$BEST_DIR" "$TRANSITION_RAW_DIR" "$TRANSITION_TRACKING_DIR"
cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[FAIL] IsaacLab Python not found: $PYTHON_BIN"
  exit 1
fi
if [[ -z "$GMR_PYTHON_BIN" ]]; then
  for candidate in \
    /home/sys01/miniconda3/envs/gmr/bin/python \
    /home/sys01/miniconda3/envs/dexjoco/bin/python \
    /data2/yangky/miniconda3/envs/gmr/bin/python; do
    if [[ -x "$candidate" ]]; then
      GMR_PYTHON_BIN="$candidate"
      break
    fi
  done
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO_ROOT/source/whole_body_tracking:$SRC_ROOT/IsaacLab/source/isaaclab:$SRC_ROOT/IsaacLab/source/isaaclab_assets:$SRC_ROOT/IsaacLab/source/isaaclab_contrib:$SRC_ROOT/IsaacLab/source/isaaclab_mimic:$SRC_ROOT/IsaacLab/source/isaaclab_rl:$SRC_ROOT/IsaacLab/source/isaaclab_tasks:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-$SRC_ROOT/tmp}"
export OMNI_USER_CACHE_DIR="${OMNI_USER_CACHE_DIR:-$TMPDIR/omni_cache}"
export OV_CACHE_ROOT="${OV_CACHE_ROOT:-$TMPDIR/ov_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$OMNI_USER_CACHE_DIR" "$OV_CACHE_ROOT"

resolve_source_manifest() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$SOURCE_MANIFEST" "$RESOLVED_MANIFEST" <<'PY'
import json
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).expanduser().resolve()
output = Path(sys.argv[3]).expanduser().resolve()
payload = json.loads(source.read_text(encoding="utf-8"))
for motion in payload["motions"]:
    for key in ("motion_file", "raw_gmr"):
        if key not in motion:
            continue
        path = Path(motion[key]).expanduser()
        motion[key] = str(path.resolve() if path.is_absolute() else (repo_root / path).resolve())
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(output)
PY
}

motion_rows() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for motion in payload["motions"]:
    print("\t".join((motion["name"], motion.get("group", ""), motion["motion_file"], motion.get("raw_gmr", ""))))
PY
}

motion_row_by_name() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target = sys.argv[2]
matches = [motion for motion in payload["motions"] if motion["name"] == target]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one motion named {target!r}; found {len(matches)}")
motion = matches[0]
print("\t".join((motion["name"], motion.get("group", ""), motion["motion_file"], motion.get("raw_gmr", ""))))
PY
}

json_value() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload.get(sys.argv[2], "")
print(value if value is not None else "")
PY
}

report_passed() {
  [[ -f "$1" ]] && [[ "$(json_value "$1" status)" == "passed" ]]
}

wait_for_gpu_memory() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi is unavailable; skipping free-memory check"
    return
  fi
  while true; do
    local free_mem
    free_mem="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    echo "[INFO] GPU${GPU} free=${free_mem}MiB required=${MIN_FREE_GPU_MEM_MB}MiB"
    if [[ "$free_mem" =~ ^[0-9]+$ ]] && (( free_mem >= MIN_FREE_GPU_MEM_MB )); then
      return
    fi
    sleep "$GPU_POLL_SECONDS"
  done
}

latest_run_dir() {
  local run_name="$1"
  find "$REPO_ROOT/logs/rsl_rl/t800_flat" -maxdepth 1 -type d -name "*_${run_name}" -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

latest_checkpoint() {
  find "$1" -maxdepth 1 -type f -name "model_*.pt" -printf "%f\n" 2>/dev/null | sort -V | tail -n 1
}

latest_existing_motion_run() {
  local motion_name="$1"
  find "$REPO_ROOT/logs/rsl_rl/t800_flat" -maxdepth 1 -type d \
    -name "*_canonical_v1_${motion_name}_r*" -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

latest_passing_training_report() {
  "$PYTHON_BIN" - "$REPORT_DIR" "$1" <<'PY'
import json
from pathlib import Path
import sys

report_dir = Path(sys.argv[1])
motion = sys.argv[2]
reports = []
for path in report_dir.glob(f"{motion}_r*_training.json"):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    if payload.get("status") == "passed" and payload.get("checkpoint"):
        reports.append((int(path.stem.split("_r")[-1].split("_")[0]), path))
if reports:
    print(max(reports, key=lambda item: item[0])[1])
PY
}

motion_round_from_run() {
  local basename
  basename="$(basename "$1")"
  if [[ "$basename" =~ _r([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  fi
}

transcode_videos() {
  local run_dir="$1"
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[WARN] ffmpeg is unavailable; videos were not normalized to H.264"
    return
  fi
  while IFS= read -r input; do
    local output="${input%.mp4}_h264.mp4"
    [[ -f "$output" ]] && continue
    ffmpeg -loglevel error -y -i "$input" -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$output" \
      || echo "[WARN] failed to transcode $input"
  done < <(find "$run_dir/videos" -type f -name '*.mp4' ! -name '*_h264.mp4' 2>/dev/null)
}

render_policy() {
  local run_dir="$1"
  local checkpoint="$2"
  local motion_file="$3"
  local label="$4"
  local log_file="$LOG_DIR/play_${label}_$(date +%Y%m%d_%H%M%S).log"
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
    --headless 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  if [[ "$status" -ne 0 ]]; then
    echo "[WARN] playback failed for $label with status $status"
  fi
  transcode_videos "$run_dir"
}

run_evaluation() {
  local run_dir="$1"
  local checkpoint="$2"
  local motion_file="$3"
  local output_report="$4"
  local label="$5"
  local log_file="$LOG_DIR/eval_${label}_$(date +%Y%m%d_%H%M%S).log"
  wait_for_gpu_memory
  "$PYTHON_BIN" "$REPO_ROOT/scripts/rsl_rl/evaluate_t800_policy.py" \
    --motion_file "$motion_file" \
    --output "$output_report" \
    --episodes "$EVAL_EPISODES" \
    --num_envs "$EVAL_ENVS" \
    --min_success_rate "$MIN_SUCCESS_RATE" \
    --load_run "$run_dir" \
    --checkpoint "$checkpoint" \
    --device "$DEVICE" \
    --headless 2>&1 | tee -a "$log_file"
  return ${PIPESTATUS[0]}
}

train_single_motion() {
  local motion_name="$1"
  local motion_file="$2"
  local max_rounds="$3"
  local best_report="$BEST_DIR/${motion_name}.json"
  if report_passed "$best_report"; then
    local saved_checkpoint
    saved_checkpoint="$(json_value "$best_report" checkpoint)"
    if [[ -f "$saved_checkpoint" ]]; then
      echo "[SKIP] $motion_name already passed: $saved_checkpoint"
      return 0
    fi
  fi

  local resume_run=""
  local resume_checkpoint=""
  local first_new_round=1

  # Re-run the physical gate for a completed checkpoint after repairing the
  # evaluator.  This preserves useful work from interrupted infrastructure.
  local prior_report
  prior_report="$(latest_passing_training_report "$motion_name")"
  if [[ -n "$prior_report" ]]; then
    local prior_run_dir prior_checkpoint prior_eval_report
    prior_run_dir="$(json_value "$prior_report" run_dir)"
    prior_checkpoint="$(basename "$(json_value "$prior_report" checkpoint)")"
    prior_eval_report="$REPORT_DIR/${motion_name}_resume_eval.json"
    echo "[EVAL] retrying saved checkpoint after evaluator repair: $prior_run_dir/$prior_checkpoint"
    if run_evaluation "$prior_run_dir" "$prior_checkpoint" "$motion_file" "$prior_eval_report" "${motion_name}_resume"; then
      render_policy "$prior_run_dir" "$prior_checkpoint" "$motion_file" "$motion_name"
      cp "$prior_eval_report" "$best_report"
      echo "[PASS] $motion_name from saved checkpoint: $prior_run_dir/$prior_checkpoint"
      return 0
    fi
  fi

  # Continue from an interrupted run even if it stopped before TensorBoard
  # acceptance could be written.
  local existing_run existing_checkpoint existing_round
  existing_run="$(latest_existing_motion_run "$motion_name")"
  if [[ -n "$existing_run" ]]; then
    existing_checkpoint="$(latest_checkpoint "$existing_run")"
    existing_round="$(motion_round_from_run "$existing_run")"
    if [[ -n "$existing_checkpoint" && -n "$existing_round" ]]; then
      resume_run="$(basename "$existing_run")"
      resume_checkpoint="$existing_checkpoint"
      first_new_round=$((existing_round + 1))
      echo "[RESUME] $motion_name from $resume_run/$resume_checkpoint; next round=$first_new_round"
    fi
  fi

  local attempt round
  for ((attempt = 1; attempt <= max_rounds; attempt++)); do
    round=$((first_new_round + attempt - 1))
    local run_name="canonical_v1_${motion_name}_r${round}"
    local train_log="$LOG_DIR/train_${run_name}_$(date +%Y%m%d_%H%M%S).log"
    local resume_args=()
    if [[ -n "$resume_run" ]]; then
      resume_args=(--resume true --load_run "$resume_run" --checkpoint "$resume_checkpoint")
    fi
    local train_overrides=()
    read -r -a train_overrides <<< "$TRAIN_OVERRIDES"

    echo "[TRAIN] motion=$motion_name round=$round attempt=$attempt/$max_rounds resume=${resume_run:-fresh}"
    wait_for_gpu_memory
    "$PYTHON_BIN" "$REPO_ROOT/scripts/rsl_rl/train_t800.py" \
      --motion_file "$motion_file" \
      --num_envs "$NUM_ENVS" \
      --max_iterations "$ITERATIONS_PER_ROUND" \
      --device "$DEVICE" \
      --run_name "$run_name" \
      --logger "$LOGGER" \
      --log_project_name "$PROJECT" \
      --task_variant "$TASK_VARIANT" \
      --headless \
      -- "${resume_args[@]}" "${train_overrides[@]}" 2>&1 | tee -a "$train_log"
    local train_status=${PIPESTATUS[0]}
    if [[ "$train_status" -ne 0 ]]; then
      echo "[FAIL] training $motion_name round $round exited with $train_status"
      return 1
    fi

    local run_dir
    run_dir="$(latest_run_dir "$run_name")"
    local checkpoint
    checkpoint="$(latest_checkpoint "$run_dir")"
    if [[ -z "$run_dir" || -z "$checkpoint" ]]; then
      echo "[FAIL] no checkpoint found for $run_name"
      return 1
    fi

    local tail_report="$REPORT_DIR/${motion_name}_r${round}_training.json"
    "$PYTHON_BIN" "$REPO_ROOT/scripts/t800_check_training_acceptance.py" \
      "$run_dir" --motion "$motion_name" --output "$tail_report"
    local tail_status=$?

    if [[ "$tail_status" -eq 0 ]]; then
      local eval_report="$REPORT_DIR/${motion_name}_r${round}_eval.json"
      run_evaluation "$run_dir" "$checkpoint" "$motion_file" "$eval_report" "$motion_name"
      local eval_status=$?
      if [[ "$eval_status" -eq 0 ]]; then
        render_policy "$run_dir" "$checkpoint" "$motion_file" "$motion_name"
        cp "$eval_report" "$best_report"
        echo "[PASS] $motion_name: $run_dir/$checkpoint"
        return 0
      fi
    elif (( attempt == max_rounds )); then
      render_policy "$run_dir" "$checkpoint" "$motion_file" "$motion_name"
    fi

    resume_run="$(basename "$run_dir")"
    resume_checkpoint="$checkpoint"
  done

  echo "[FAIL] $motion_name did not reach the acceptance gate in $max_rounds rounds"
  return 1
}

convert_transition_references() {
  "$PYTHON_BIN" "$REPO_ROOT/scripts/t800_build_stand_action_sequences.py" \
    --manifest "$RESOLVED_MANIFEST" \
    --repo-root "$REPO_ROOT" \
    --output-dir "$TRANSITION_RAW_DIR" \
    --tracking-output-dir "$TRANSITION_TRACKING_DIR" \
    --output-manifest "$TRANSITION_MANIFEST"

  if [[ -x "$GMR_PYTHON_BIN" && -d "$SRC_ROOT/gmr" ]]; then
    local raw_inputs=()
    while IFS=$'\t' read -r _ _ _ raw_file; do
      raw_inputs+=("$raw_file")
    done < <(motion_rows "$TRANSITION_MANIFEST")
    echo "[RENDER] stand/action/stand references -> $TRANSITION_REVIEW_DIR"
    PYTHON_BIN="$GMR_PYTHON_BIN" GMR_ROOT="$SRC_ROOT/gmr" OUTPUT_ROOT="$TRANSITION_REVIEW_DIR" \
      MAX_SECONDS=10 VIEWS=front,right "$REPO_ROOT/scripts/t800_render_reference_review.sh" "${raw_inputs[@]}" \
      || echo "[WARN] transition reference rendering failed"
  else
    echo "[WARN] GMR environment is unavailable; transition reference rendering skipped"
  fi

  while IFS=$'\t' read -r motion_name _ tracking_file raw_file; do
    if [[ ! -f "$tracking_file" ]]; then
      echo "[CONVERT] $motion_name -> $tracking_file"
      wait_for_gpu_memory
      "$PYTHON_BIN" "$REPO_ROOT/scripts/t800_csv_to_npz.py" \
        --input_file "$raw_file" \
        --input_format gmr_npz \
        --output_name "${motion_name}_stand_action_stand_tracking" \
        --output_path "$tracking_file" \
        --output_fps 50 \
        --skip_wandb_upload \
        --headless \
        --device "$DEVICE" || return 1
    fi
    "$PYTHON_BIN" "$REPO_ROOT/scripts/t800_validate_joint_contract.py" "$tracking_file" --raw-gmr "$raw_file" \
      || return 1
  done < <(motion_rows "$TRANSITION_MANIFEST")
}

write_joint_best_report() {
  local run_dir="$1"
  local checkpoint="$2"
  shift 2
  "$PYTHON_BIN" - "$BEST_DIR/joint_standing5.json" "$run_dir" "$checkpoint" "$@" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
payload = {
    "status": "passed",
    "run_dir": sys.argv[2],
    "checkpoint": str(Path(sys.argv[2]) / sys.argv[3]),
    "per_motion_reports": sys.argv[4:],
}
output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

train_joint_policy() {
  local seed_report="$BEST_DIR/straight_punch.json"
  local resume_checkpoint
  resume_checkpoint="$(json_value "$seed_report" checkpoint)"
  local resume_run
  resume_run="$(basename "$(dirname "$resume_checkpoint")")"
  resume_checkpoint="$(basename "$resume_checkpoint")"

  local round
  for ((round = 1; round <= JOINT_MAX_ROUNDS; round++)); do
    local run_name="canonical_v1_joint_standing5_r${round}"
    local train_log="$LOG_DIR/train_${run_name}_$(date +%Y%m%d_%H%M%S).log"
    local train_overrides=()
    read -r -a train_overrides <<< "$TRAIN_OVERRIDES"
    echo "[TRAIN] joint standing5 round=$round/$JOINT_MAX_ROUNDS resume=$resume_run/$resume_checkpoint"
    wait_for_gpu_memory
    "$PYTHON_BIN" "$REPO_ROOT/scripts/rsl_rl/train_t800.py" \
      --motion_file "$TRANSITION_MANIFEST" \
      --num_envs "$NUM_ENVS" \
      --max_iterations "$ITERATIONS_PER_ROUND" \
      --device "$DEVICE" \
      --run_name "$run_name" \
      --logger "$LOGGER" \
      --log_project_name "$PROJECT" \
      --task_variant "$TASK_VARIANT" \
      --headless \
      -- --resume true --load_run "$resume_run" --checkpoint "$resume_checkpoint" \
      "${train_overrides[@]}" 2>&1 | tee -a "$train_log"
    local train_status=${PIPESTATUS[0]}
    [[ "$train_status" -eq 0 ]] || return "$train_status"

    local run_dir
    run_dir="$(latest_run_dir "$run_name")"
    local checkpoint
    checkpoint="$(latest_checkpoint "$run_dir")"
    [[ -n "$run_dir" && -n "$checkpoint" ]] || return 1

    local tail_report="$REPORT_DIR/joint_standing5_r${round}_training.json"
    "$PYTHON_BIN" "$REPO_ROOT/scripts/t800_check_training_acceptance.py" \
      "$run_dir" --motion joint_standing5 --output "$tail_report"
    local tail_status=$?
    if [[ "$tail_status" -eq 0 ]]; then
      local all_passed=1
      local eval_reports=()
      while IFS=$'\t' read -r motion_name _ tracking_file _; do
        local eval_report="$REPORT_DIR/joint_standing5_r${round}_${motion_name}_eval.json"
        run_evaluation "$run_dir" "$checkpoint" "$tracking_file" "$eval_report" "joint_${motion_name}"
        local eval_status=$?
        eval_reports+=("$eval_report")
        [[ "$eval_status" -eq 0 ]] || all_passed=0
      done < <(motion_rows "$TRANSITION_MANIFEST")
      if [[ "$all_passed" -eq 1 ]]; then
        local switch_names=()
        while IFS=$'\t' read -r motion_name _ _ _; do
          switch_names+=("$motion_name")
        done < <(motion_rows "$TRANSITION_MANIFEST")
        wait_for_gpu_memory
        "$PYTHON_BIN" "$REPO_ROOT/scripts/rsl_rl/play_t800_policy_switch.py" \
          --load_run "$run_dir" \
          --checkpoint "$checkpoint" \
          --motion_file "$TRANSITION_MANIFEST" \
          --switch_motions "${switch_names[@]}" \
          --switch_interval 450 \
          --no_snap_robot_on_switch \
          --video \
          --video_length 2250 \
          --num_envs 5 \
          --device "$DEVICE" \
          --video_folder_name qualifier_joint_switch \
          --headless
        transcode_videos "$run_dir"
        write_joint_best_report "$run_dir" "$checkpoint" "${eval_reports[@]}"
        echo "[PASS] joint standing5 policy: $run_dir/$checkpoint"
        return 0
      fi
    fi

    resume_run="$(basename "$run_dir")"
    resume_checkpoint="$checkpoint"
  done
  echo "[FAIL] joint standing5 policy did not pass all five per-motion gates"
  return 1
}

echo "[INFO] staged queue started at $(date --iso-8601=seconds)"
echo "[INFO] source manifest: $SOURCE_MANIFEST"
echo "[INFO] state root: $STATE_ROOT"
resolve_source_manifest

while IFS=$'\t' read -r motion_name _ motion_file raw_file; do
  [[ -f "$motion_file" ]] || { echo "[FAIL] missing canonical NPZ: $motion_file"; exit 1; }
  "$PYTHON_BIN" "$REPO_ROOT/scripts/t800_validate_joint_contract.py" "$motion_file" --raw-gmr "$raw_file" \
    || exit 1
done < <(motion_rows "$RESOLVED_MANIFEST")

standing_failures=0
IFS=',' read -r -a standing_motion_names <<< "$STANDING_MOTION_ORDER"
if [[ "${#standing_motion_names[@]}" -ne 5 ]]; then
  echo "[FAIL] STANDING_MOTION_ORDER must contain all five standing motions; got: $STANDING_MOTION_ORDER"
  exit 1
fi
declare -A seen_standing_motion_names=()
for selected_motion in "${standing_motion_names[@]}"; do
  if [[ -n "${seen_standing_motion_names[$selected_motion]:-}" ]]; then
    echo "[FAIL] duplicate motion in STANDING_MOTION_ORDER: $selected_motion"
    exit 1
  fi
  seen_standing_motion_names["$selected_motion"]=1
  selected_row="$(motion_row_by_name "$RESOLVED_MANIFEST" "$selected_motion")" || exit 1
  IFS=$'\t' read -r motion_name group motion_file _ <<< "$selected_row"
  if [[ "$group" != "standing_action" ]]; then
    echo "[FAIL] $motion_name is not a standing_action motion"
    exit 1
  fi
  train_single_motion "$motion_name" "$motion_file" "$SINGLE_MAX_ROUNDS" || standing_failures=$((standing_failures + 1))
done

while IFS=$'\t' read -r motion_name group motion_file _; do
  [[ "$group" == "recovery" ]] || continue
  train_single_motion "$motion_name" "$motion_file" "$RECOVERY_MAX_ROUNDS" \
    || echo "[WARN] recovery remains independent and has not passed yet"
done < <(motion_rows "$RESOLVED_MANIFEST")

if (( standing_failures > 0 )); then
  echo "[BLOCKED] $standing_failures standing motion(s) failed; joint training was not started"
  exit 2
fi

convert_transition_references || exit 1
train_joint_policy || exit 2

echo "[OK] all staged T800 qualifier training gates passed"
