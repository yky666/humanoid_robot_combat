# humanoid_robot_combat

`humanoid_robot_combat` is our Isaac Lab based humanoid combat-motion training stack. It extends the original
`whole_body_tracking` / BeyondMimic motion tracking codebase with:

- T800 humanoid support for tracking training and playback
- local-motion training without a mandatory W&B motion registry
- a practical retargeting pipeline from public motion datasets to tracking-ready T800 motions
- convenience scripts for conversion, smoke testing, training, and video recording
- extra robot/config scaffolding for Pi-Plus and PM01 experiments

The current focus of this repository is **combat-style motion learning**, especially boxing and kicking sequences that
can be retargeted onto T800 and trained with PPO in Isaac Lab.

## What Is Included

- **Tracking environments**
  - `Tracking-Flat-T800-v0`
  - `Tracking-Flat-T800-Low-Freq-v0`
  - `Tracking-Flat-T800-Wo-State-Estimation-v0`
- **Robot definitions**
  - `T800`
  - `Pi-Plus`
  - `PM01`
- **Utility scripts**
  - `scripts/t800_csv_to_npz.py`
  - `scripts/check_npz.py`
  - `scripts/batch_prepare_t800_motions.py`
  - `scripts/batch_train_t800.py`
  - `scripts/rsl_rl/train_t800.py`
  - `scripts/rsl_rl/play_t800.py`

## Repository Layout

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t800`
  T800 tracking environment, MDP, and PPO configuration.
- `source/whole_body_tracking/whole_body_tracking/robots/t800.py`
  T800 robot description and control parameters used by Isaac Lab.
- `scripts/t800_csv_to_npz.py`
  Converts T800 motion CSV, GMR pickle output, or legacy 40-column local numpy motions into tracking-ready `.npz`.
- `scripts/check_npz.py`
  Verifies that a generated motion file contains the required keys for the motion loader.
- `configs/t800_combat_motion_manifest.json`
  Canonical list of combat motion inputs, output targets, and default run names.
- `scripts/batch_prepare_t800_motions.py`
  Batch converts manifest-listed motions into tracking-ready `.npz`.
- `scripts/batch_train_t800.py`
  Batch launches smoke or formal training runs from the manifest.
- `scripts/rsl_rl/train_t800.py`
  Thin wrapper around the generic trainer with T800-friendly defaults.
- `scripts/rsl_rl/play_t800.py`
  Thin wrapper around the generic player with T800-friendly defaults.
- `artifacts/boxing_T800`
  Local motion assets and generated T800 combat motions. This directory is intentionally ignored in git.

## Environment Setup

We currently use two conda environments:

- `gmr`
  Used for SMPL-X / AMASS motion retargeting with General Motion Retargeting.
- `env_isaaclab`
  Used for Isaac Lab, motion replay, training, smoke tests, and play-video recording.

See [QUICKSTART.md](QUICKSTART.md) for the end-to-end workflow.

## Motion Pipeline

The main public-data pipeline we use for T800 combat motions is:

```text
AMASS / SMPL-X motion
  -> GMR retargeting to T800
  -> GMR pickle (.pkl or pickle-backed .npz)
  -> CSV
  -> tracking-ready T800 NPZ
  -> smoke test
  -> formal PPO training
  -> play video recording
```

For legacy BVH motions, we also support:

```text
.bvh -> GMR -> .pkl -> CSV -> tracking-ready T800 NPZ
```

For legacy local T800 arrays, we also support:

```text
.npy (40 columns) -> tracking-ready T800 NPZ
```

## Typical Commands

### 1. Retarget a motion with GMR

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate gmr
cd /data2/yangky/test/GMR

python scripts/smplx_to_robot.py \
  --smplx_file /data2/yangky/test/datasets/AMASS/CMU/CMU/13/13_18_stageii.npz \
  --robot t800 \
  --save_path /data2/yangky/test/GMR/output/cmu13_18_t800.pkl
```

### 2. Convert GMR output to CSV

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate gmr
cd /data2/yangky/test/GMR

python scripts/batch_gmr_pkl_to_csv.py \
  --folder /data2/yangky/test/GMR/output
```

### 3. Convert CSV to tracking NPZ

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

Batch prepare from the tracked manifest:

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

python scripts/batch_prepare_t800_motions.py \
  --groups accad_combat_batch legacy_local
```

### 4. Validate the generated NPZ

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

python scripts/check_npz.py \
  /data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/cmu_boxing_batch/cmu13_18_t800_tracking.npz
```

### 5. Smoke test training

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

Manifest-driven smoke batch:

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

### 6. Formal training

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/train_t800.py \
  --motion_file /data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/cmu_boxing_batch/cmu13_18_t800_tracking.npz \
  --num_envs 1024 \
  --max_iterations 30000 \
  --headless \
  --device cuda:0 \
  --run_name cmu13_18_t800_v1 \
  --logger wandb \
  --log_project_name t800_boxing
```

Manifest-driven formal batch:

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

### 7. Resume from a stable checkpoint

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

### 8. Record a play video

```bash
source /data2/yangky/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /data2/yangky/test/whole_body_tracking

CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/play_t800.py \
  --load_run 2026-04-21_20-20-54_cmu15_13_t800_v1_from29999 \
  --checkpoint model_59998.pt \
  --motion_file /data2/yangky/test/whole_body_tracking/artifacts/boxing_T800/cmu_boxing_batch/cmu15_13_t800_tracking.npz \
  --num_envs 25 \
  --video \
  --video_length 800 \
  --headless \
  --device cuda:0
```

## Current Motion Selection Strategy

We recommend the following motion-selection funnel:

1. Prefer public datasets with explicit combat semantics:
   `boxing`, `jab`, `cross`, `hook`, `uppercut`, `front_kick`, `roundhouse`, `push_kick`.
2. Retarget the motion to T800 through GMR.
3. Convert to tracking-ready `.npz` and run `scripts/check_npz.py`.
4. Run a one-iteration smoke test with `64` environments.
5. Rank candidates by:
   - mean reward
   - mean episode length
   - `Metrics/motion/error_body_pos`
   - `Metrics/motion/error_joint_pos`
   - termination dominated by `ee_body_pos` or not
6. Only launch formal training for motions that are stable under smoke.

## Notes

- Generated `.csv`, `.pkl`, `.npz`, `wandb`, `logs`, `videos`, and `artifacts` are intentionally ignored in git.
- This repository contains project-specific engineering for T800 combat training. It is built on top of the original
  `whole_body_tracking` / BeyondMimic motion tracking framework.

## License

This repository is released under the MIT license. See [LICENSE](LICENSE).
