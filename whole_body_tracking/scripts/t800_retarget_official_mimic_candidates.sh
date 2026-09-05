#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_ROOT="${SRC_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
GMR_ROOT="${GMR_ROOT:-$SRC_ROOT/gmr}"
PYTHON_BIN="${PYTHON_BIN:-/data2/yangky/miniconda3/envs/gmr/bin/python}"
AMASS_ACCAD_ROOT="${AMASS_ACCAD_ROOT:-$SRC_ROOT/datasets/AMASS/ACCAD/ACCAD}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/official_mimic_T800}"
STAGE_SRC="${STAGE_SRC:-$ARTIFACT_ROOT/smplx_src}"
PKL_ROOT="${PKL_ROOT:-$ARTIFACT_ROOT/gmr_pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/review_videos/t800_reference_audit_20260902/official_mimic_candidates}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
NUM_CPUS="${NUM_CPUS:-1}"
MEMORY_THRESHOLD_GB="${MEMORY_THRESHOLD_GB:-5}"
RENDER_FPS="${RENDER_FPS:-25}"
MAX_SECONDS="${MAX_SECONDS:-0}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
VIEWS="${VIEWS:-front,right,quarter}"
OVERWRITE="${OVERWRITE:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[FAIL] missing python: $PYTHON_BIN"
  exit 1
fi
if [[ ! -d "$GMR_ROOT" ]]; then
  echo "[FAIL] missing GMR_ROOT: $GMR_ROOT"
  exit 1
fi
if [[ ! -d "$AMASS_ACCAD_ROOT" ]]; then
  echo "[FAIL] missing AMASS ACCAD root: $AMASS_ACCAD_ROOT"
  exit 1
fi

mkdir -p "$STAGE_SRC" "$PKL_ROOT" "$OUTPUT_ROOT"

link_source() {
  local src="$1"
  local dst_name="$2"
  if [[ ! -f "$src" ]]; then
    echo "[FAIL] missing AMASS source: $src"
    exit 1
  fi
  ln -sfn "$src" "$STAGE_SRC/$dst_name"
}

link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz" \
  "punch_jab_left_e1_stageii.npz"
link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsPunches_c3d/E2_-__Jab_right_stageii.npz" \
  "punch_jab_right_e2_stageii.npz"
link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsPunches_c3d/E3_-__cross_left_stageii.npz" \
  "punch_cross_left_e3_stageii.npz"
link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsPunches_c3d/E4_-__cross_right_stageii.npz" \
  "punch_cross_right_e4_stageii.npz"
link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsPunches_c3d/E5_-__hook_left_stageii.npz" \
  "punch_hook_left_e5_stageii.npz"
link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsPunches_c3d/E6_-__hook_right_stageii.npz" \
  "punch_hook_right_e6_stageii.npz"
link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsKicks_c3d/G3_-_front_kick_stageii.npz" \
  "kick_front_g3_stageii.npz"
link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsKicks_c3d/G17-__push_kick_left_stageii.npz" \
  "kick_push_left_g17_stageii.npz"
link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsKicks_c3d/G18-__push_kick_right_stageii.npz" \
  "kick_push_right_g18_stageii.npz"
link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsKicks_c3d/G4_-spinning_back_kick_stageii.npz" \
  "kick_spinning_back_g4_stageii.npz"
link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz" \
  "kick_roundhouse_left_g8_stageii.npz"
link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsKicks_c3d/G9_-__roundhouse_right_stageii.npz" \
  "kick_roundhouse_right_g9_stageii.npz"
link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsKicks_c3d/G19-__reverse_spin_cresent_left_stageii.npz" \
  "kick_reverse_spin_cresent_left_g19_stageii.npz"
link_source "$AMASS_ACCAD_ROOT/Male2MartialArtsKicks_c3d/G20_-__reverse_spin_cresent_right_stageii.npz" \
  "kick_reverse_spin_cresent_right_g20_stageii.npz"

find "$STAGE_SRC" -maxdepth 1 -type l -name '*.npz' | sort > "$OUTPUT_ROOT/source_symlinks.txt"

cd "$GMR_ROOT"
echo "[INFO] retargeting official mimic candidates to T800"
echo "[INFO] source: $STAGE_SRC"
echo "[INFO] pkl output: $PKL_ROOT"
env PYTHONPATH="$GMR_ROOT:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="$GPU_ID" \
  "$PYTHON_BIN" scripts/smplx_to_robot_dataset.py \
  --robot t800 \
  --src_folder "$STAGE_SRC" \
  --tgt_folder "$PKL_ROOT" \
  --num_cpus "$NUM_CPUS" \
  --device "$DEVICE" \
  --memory_threshold_gb "$MEMORY_THRESHOLD_GB" \
  --override

mapfile -t pkl_inputs < <(find "$PKL_ROOT" -type f -name '*.pkl' | sort)
if [[ "${#pkl_inputs[@]}" -eq 0 ]]; then
  echo "[FAIL] no retargeted pkl files found in $PKL_ROOT"
  exit 1
fi
printf '%s\n' "${pkl_inputs[@]}" > "$OUTPUT_ROOT/retargeted_pkl_inputs.txt"

render_args=(
  --input-root "$PKL_ROOT"
  --input-format pkl
  --output-root "$OUTPUT_ROOT"
  --render-fps "$RENDER_FPS"
  --max-seconds "$MAX_SECONDS"
  --width "$WIDTH"
  --height "$HEIGHT"
  --views "$VIEWS"
  --index-file "$OUTPUT_ROOT/index.html"
)
if [[ "$OVERWRITE" == "1" ]]; then
  render_args+=(--overwrite)
fi
for input in "${pkl_inputs[@]}"; do
  render_args+=(--input-file "$input")
done

echo "[INFO] rendering ${#pkl_inputs[@]} official mimic candidate(s)"
echo "[INFO] video output: $OUTPUT_ROOT"
exec env PYTHONPATH="$GMR_ROOT:${PYTHONPATH:-}" MUJOCO_GL="${MUJOCO_GL:-egl}" \
  "$PYTHON_BIN" scripts/render_motiondecode_t800_videos.py "${render_args[@]}"
