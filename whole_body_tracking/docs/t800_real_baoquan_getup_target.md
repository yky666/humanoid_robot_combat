# T800 Real Baoquan Terminal Target

Date: 2026-09-07

This note records how the real-robot `pdstand2baoquan` joint log is turned into a measured boxing-ready terminal target for direct-RL get-up training.

## Source Data

The source archive is outside the repository:

```bash
/mnt/data/yangky/test/datasets/urkl_locomotion_260901/motion_logs.zip
```

Relevant members:

- `logs/pdstand2baoquan.csv`: measured transition from PD stand to boxing guard.
- `logs/baoquan2pdstand.csv`: reverse transition for comparison.
- `logs/prone2pdstand.csv`: real-robot prone recovery joint-only log.
- `logs/supine2pdstand.csv`: real-robot supine recovery joint-only log.

The CSV logs contain `t_host`, 25 position columns, 25 velocity columns, and 25 torque columns in EngineAI SDK semantic order. The right arm and head numeric names differ from the policy/MuJoCo names, so the conversion maps SDK order to canonical T800 policy order.

## Extract The Measured Target

```bash
python whole_body_tracking/scripts/t800_extract_real_baoquan_pose.py \
  --input /mnt/data/yangky/test/datasets/urkl_locomotion_260901/motion_logs.zip \
  --member logs/pdstand2baoquan.csv \
  --tail-seconds 1.0 \
  --output-json results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json \
  --output-npz results/t800_real_baoquan_getup_20260907/measured_boxing_ready.npz
```

Current extracted target:

- frames: `4885`
- duration: `9.749307 s`
- estimated sampling rate: `500.250 Hz`
- tail window: `500` frames
- maximum tail joint-position std: `0.000369 rad`
- mean tail joint-position std: `0.000039 rad`

The low tail variance means the final guard pose is stable enough to use as a terminal-pose candidate.

## Render The Joint Log In MuJoCo

```bash
python whole_body_tracking/scripts/t800_render_real_joint_log_mujoco.py \
  --input /mnt/data/yangky/test/datasets/urkl_locomotion_260901/motion_logs.zip \
  --member logs/pdstand2baoquan.csv \
  --output results/t800_real_baoquan_getup_20260907/pdstand2baoquan_mujoco_front_right.mp4 \
  --metadata-json results/t800_real_baoquan_getup_20260907/pdstand2baoquan_mujoco_front_right.json \
  --views front,right \
  --render-fps 30 \
  --width 640 \
  --height 480 \
  --tail-hold-seconds 1.5 \
  --floor-align each
```

The render is a kinematic joint-pose inspection. The source logs do not contain root pose or IMU orientation, so the script fixes the root upright and aligns the lowest geometry to the floor.

Generated review video:

```bash
results/t800_real_baoquan_getup_20260907/pdstand2baoquan_mujoco_front_right.mp4
```

## Train With The Measured Target

Short smoke checks:

```bash
cd whole_body_tracking

conda run -n env_isaaclab python scripts/rsl_rl/train_t800.py \
  --task_variant getup_supine \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json \
  --num_envs 32 \
  --max_iterations 2 \
  --device cuda:0 \
  --run_name t800_getup_supine_measured_baoquan_smoke \
  --logger tensorboard \
  --headless

conda run -n env_isaaclab python scripts/rsl_rl/train_t800.py \
  --task_variant getup_prone \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json \
  --num_envs 32 \
  --max_iterations 2 \
  --device cuda:0 \
  --run_name t800_getup_prone_measured_baoquan_smoke \
  --logger tensorboard \
  --headless
```

Longer training should be launched only after the MuJoCo review video is accepted:

```bash
cd whole_body_tracking

conda run -n env_isaaclab python scripts/rsl_rl/train_t800.py \
  --task_variant getup_mixed \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json \
  --num_envs 1024 \
  --max_iterations 200 \
  --device cuda:0 \
  --run_name t800_getup_mixed_measured_baoquan_v1 \
  --logger tensorboard \
  --headless
```

For playback/evaluation, pass the same `--getup_target_json` to keep the policy target and success criteria aligned.
