# Quick Start

This quick start documents the current working workflow for T800 combat-motion experiments.

## 1. Conda Environments

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate gmr
```

Use `gmr` for:

- `smplx_to_robot.py`
- `bvh_to_robot.py`
- `batch_gmr_pkl_to_csv.py`

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

Use `env_isaaclab` for:

- `t800_csv_to_npz.py`
- `batch_prepare_t800_motions.py`
- `check_npz.py`
- `batch_train_t800.py`
- `train_t800.py`
- `play_t800.py`

## 2. Recommended Data Sources

### CMU Boxing

- `CMU/13/13_17_stageii.npz`
- `CMU/13/13_18_stageii.npz`
- `CMU/15/15_13_stageii.npz`
- `CMU/17/17_10_stageii.npz`

### ACCAD Martial Arts

Punches:

- `Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz`
- `Male2MartialArtsPunches_c3d/E2_-__Jab_right_stageii.npz`
- `Male2MartialArtsPunches_c3d/E5_-__hook_left_stageii.npz`
- `Male2MartialArtsPunches_c3d/E6_-__hook_right_stageii.npz`
- `Male2MartialArtsPunches_c3d/E7_-_uppercut_left_stageii.npz`
- `Male2MartialArtsPunches_c3d/E8_-_uppercut_right_stageii.npz`

Kicks:

- `Male2MartialArtsKicks_c3d/G3_-_front_kick_stageii.npz`
- `Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz`
- `Male2MartialArtsKicks_c3d/G9_-__roundhouse_right_stageii.npz`
- `Male2MartialArtsKicks_c3d/G17-__push_kick_left_stageii.npz`
- `Male2MartialArtsKicks_c3d/G18-__push_kick_right_stageii.npz`

## 3. Retarget AMASS Motion to T800

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate gmr
cd /data2/yangky/test/GMR

python scripts/smplx_to_robot.py \
  --smplx_file /data2/yangky/test/datasets/AMASS/CMU/CMU/13/13_18_stageii.npz \
  --robot t800 \
  --save_path /data2/yangky/test/GMR/output/cmu13_18_t800.pkl
```

For a BVH input:

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate gmr
cd /data2/yangky/test/GMR

xvfb-run -a python scripts/bvh_to_robot.py \
  --bvh_file /path/to/motion.bvh \
  --robot t800 \
  --format lafan1 \
  --save_path /data2/yangky/test/GMR/output/my_motion.pkl
```

## 4. Convert GMR Pickles to CSV

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate gmr
cd /data2/yangky/test/GMR

python scripts/batch_gmr_pkl_to_csv.py \
  --folder /data2/yangky/test/GMR/output
```

## 5. Convert CSV to Tracking NPZ

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

python scripts/t800_csv_to_npz.py \
  --input_file /data2/yangky/test/GMR/output/csv/cmu13_18_t800.csv \
  --input_fps 120 \
  --output_name cmu13_18_t800_tracking \
  --output_fps 50 \
  --headless
```

For a legacy local `.npy` motion:

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

python scripts/t800_csv_to_npz.py \
  --input_file /data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/riot_combo.npy \
  --input_format legacy_npy \
  --input_fps 50 \
  --output_name riot_combo_tracking \
  --output_path /data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/riot_combo_tracking.npz \
  --output_fps 50 \
  --headless
```

To prepare a whole candidate set described in `configs/t800_combat_motion_manifest.json`:

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

python scripts/batch_prepare_t800_motions.py \
  --groups accad_combat_batch legacy_local
```

Recommended destination for generated T800 motions:

```text
/data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/
/data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/cmu_boxing_batch/
```

Avoid `/tmp` for long-lived motion assets.

## 6. Validate Motion Files

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

python scripts/check_npz.py \
  /data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/cmu_boxing_batch/cmu13_18_t800_tracking.npz
```

Expected required keys:

- `fps`
- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

## 6.5 Pad Very Short Motions

Some strike-only motions are extremely short after retargeting. If you want the
robot to hold the finishing pose a little longer, you can append repeated end
frames:

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

python scripts/pad_motion.py \
  --input_file /data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/accad_combat_batch/accad_e1_jab_left_t800_tracking.npz \
  --output_file /data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/accad_combat_batch/accad_e1_jab_left_t800_tracking_padded.npz \
  --pad_frames 100
```

## 7. Smoke Test

Use a minimal one-iteration run first.

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/train_t800.py \
  --motion_file /data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/cmu_boxing_batch/cmu13_18_t800_tracking.npz \
  --num_envs 64 \
  --max_iterations 1 \
  --headless \
  --device cuda:0 \
  --run_name smoke_cmu13_18_t800 \
  --logger wandb \
  --log_project_name t800_boxing
```

Check these metrics before moving on:

- `Mean reward`
- `Mean episode length`
- `Metrics/motion/error_body_pos`
- `Metrics/motion/error_joint_pos`
- `Episode_Termination/ee_body_pos`

Batch smoke entrypoint:

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

CUDA_VISIBLE_DEVICES=1 python scripts/batch_train_t800.py \
  --stage smoke \
  --groups accad_combat_batch legacy_local \
  --gpu 1 \
  --device cuda:0
```

## 8. Formal Training

### Fresh training

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/train_t800.py \
  --motion_file /data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/cmu_boxing_batch/cmu13_17_t800_tracking.npz \
  --num_envs 1024 \
  --max_iterations 30000 \
  --headless \
  --device cuda:0 \
  --run_name cmu13_17_t800_v1 \
  --logger wandb \
  --log_project_name t800_boxing
```

### Resume from a stable checkpoint

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/train_t800.py \
  --motion_file /data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/cmu_boxing_batch/cmu15_13_t800_tracking.npz \
  --num_envs 1024 \
  --max_iterations 90000 \
  --headless \
  --device cuda:0 \
  --run_name cmu15_13_t800_v1_from29999 \
  --logger wandb \
  --log_project_name t800_boxing \
  -- --resume true \
     --load_run 2026-04-17_12-51-22_cmu13_18_t800_v2_from1499 \
     --checkpoint model_29999.pt
```

Batch formal entrypoint:

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

CUDA_VISIBLE_DEVICES=1 python scripts/batch_train_t800.py \
  --stage formal \
  --groups accad_combat_batch \
  --gpu 1 \
  --device cuda:0
```

### Run in tmux

```bash
tmux new -s boxing
```

Then start training inside the session, or launch non-interactively with `tmux new -d -s ...`.

## 9. Record Play Videos

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/play_t800.py \
  --load_run 2026-04-21_20-32-44_cmu13_17_t800_v1_from29999 \
  --checkpoint model_53900.pt \
  --motion_file /data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/cmu_boxing_batch/cmu13_17_t800_tracking.npz \
  --num_envs 25 \
  --video \
  --video_length 800 \
  --headless \
  --device cuda:0
```

The recorded videos land under:

```text
logs/rsl_rl/t800_flat/<run_name>/videos/play/
```

## 10. Current Stable T800 Baseline

The current T800 baseline was stabilized around:

- `bridge_frames = 35`
- `pd_stand_reset_ratio = 0.35`
- `init_noise_std = 0.6`
- `entropy_coef = 0.01`
- `learning_rate = 3e-4`
- `desired_kl = 0.02`
- `num_steps_per_env = 32`

This is the best current starting point for new combat motions.
