#!/usr/bin/env bash
# Render FG72 baoquan policy with fixed virtual-stick commands.
set -euo pipefail

ROOT=/mnt/data/yangky/test/humanoid_robot_combat/whole_body_tracking
PY=/home/sys01/miniconda3/envs/env_isaaclab/bin/python
OUT=/mnt/data/yangky/test/humanoid_robot_combat/whole_body_tracking/results/t800_fixed_guard72_20260910/pipeline/videos/omni_dirs
LOAD_RUN=2026-09-11_05-01-45_t800_fg72_auto_s5_r3_050130
CKPT=model_7500.pt
TASK=Tracking-Rough-T800-Fixed-Guard-72-Play-v0
VIDEO_LEN="${VIDEO_LEN:-400}"

mkdir -p "$OUT"
source /home/sys01/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# name vx vy wz
DIRS=(
  "stand 0.00 0.00 0.00"
  "fwd 0.30 0.00 0.00"
  "back -0.25 0.00 0.00"
  "left 0.00 0.30 0.00"
  "right 0.00 -0.30 0.00"
  "yaw_ccw 0.00 0.00 0.40"
  "yaw_cw 0.00 0.00 -0.40"
  "omni 0.25 0.20 0.30"
)

for spec in "${DIRS[@]}"; do
  read -r name vx vy wz <<<"$spec"
  dest="$OUT/${name}.mp4"
  log="$OUT/${name}.log"
  echo "[INFO] $(date +%H:%M:%S) render $name vx=$vx vy=$vy wz=$wz"
  FG_PLAY_VX="$vx" FG_PLAY_VY="$vy" FG_PLAY_WZ="$wz" \
    "$PY" scripts/rsl_rl/play.py \
      --task "$TASK" \
      --num_envs 4 \
      --device cuda:0 \
      --load_run "$LOAD_RUN" \
      --checkpoint "$CKPT" \
      --video \
      --video_length "$VIDEO_LEN" \
      --headless \
      --enable_cameras \
      >"$log" 2>&1 || { echo "[ERR] $name failed, see $log"; continue; }
  latest=$(ls -t logs/rsl_rl/t800_fixed_guard_velocity_72d/"$LOAD_RUN"/videos/play/*.mp4 2>/dev/null | head -n 1 || true)
  if [ -n "$latest" ]; then
    cp -f "$latest" "$dest"
    echo "[OK] $dest ($(stat -c%s "$dest") bytes)"
  else
    echo "[ERR] no mp4 for $name"
  fi
done

echo "[DONE] $(date) videos in $OUT"
ls -l "$OUT"/*.mp4 2>/dev/null || true
