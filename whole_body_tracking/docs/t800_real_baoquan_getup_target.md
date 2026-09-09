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
  --output-npz results/t800_real_baoquan_getup_20260907/measured_boxing_ready.npz \
  --output-tail-npz results/t800_real_baoquan_getup_20260907/baoquan_tail_reference_1s.npz
```

Current extracted target:

- frames: `4885`
- duration: `9.749307 s`
- estimated sampling rate: `500.250 Hz`
- tail window: `500` frames
- maximum tail joint-position std: `0.000369 rad`
- mean tail joint-position std: `0.000039 rad`

The low tail variance means the final guard pose is stable enough to use as a terminal-pose candidate.

For a small real-motion reference action, the approved run also extracts a
longer stable hold:

```bash
python whole_body_tracking/scripts/t800_extract_real_baoquan_pose.py \
  --input /mnt/data/yangky/test/datasets/urkl_locomotion_260901/motion_logs.zip \
  --member logs/pdstand2baoquan.csv \
  --tail-seconds 2.0 \
  --output-json results/t800_real_baoquan_getup_20260907/measured_boxing_ready_2s_tail.json \
  --output-npz results/t800_real_baoquan_getup_20260907/measured_boxing_ready_2s_tail.npz \
  --output-tail-npz results/t800_real_baoquan_getup_20260907/baoquan_tail_reference_2s.npz
```

`baoquan_tail_reference_2s.npz` contains `joint_pos`, `joint_vel`, `joint_tau`,
`time_s`, `fps`, `joint_names`, and the median `target_joint_pos`, all in
canonical T800 policy order.

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

Longer training after the MuJoCo review video is accepted:

```bash
cd whole_body_tracking

conda run -n env_isaaclab python scripts/rsl_rl/train_t800.py \
  --task_variant getup_supine \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json \
  --num_envs 1024 \
  --max_iterations 200 \
  --device cuda:0 \
  --run_name t800_getup_supine_measured_baoquan_v1 \
  --logger tensorboard \
  --headless

conda run -n env_isaaclab python scripts/rsl_rl/train_t800.py \
  --task_variant getup_prone \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json \
  --num_envs 1024 \
  --max_iterations 200 \
  --device cuda:1 \
  --run_name t800_getup_prone_measured_baoquan_v1 \
  --logger tensorboard \
  --headless
```

For playback/evaluation, pass the same `--getup_target_json` to keep the policy target and success criteria aligned.

## Shaped Direct-RL Pilot

The direct v1 tasks were too sparse for cold-start prone/supine get-up. The
shaped variants add broad upright/height progress rewards and keep head contact
as a penalty instead of an early termination:

- `getup_prone_shaped` -> `Getup-Direct-T800-Prone-Shaped-v0`
- `getup_supine_shaped` -> `Getup-Direct-T800-Supine-Shaped-v0`
- `getup_mixed_shaped` -> `Getup-Direct-T800-Mixed-Shaped-v0`

Example training commands:

```bash
cd whole_body_tracking

conda run -n env_isaaclab python scripts/rsl_rl/train_t800.py \
  --task_variant getup_prone_shaped \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json \
  --num_envs 1024 \
  --max_iterations 500 \
  --device cuda:1 \
  --run_name t800_getup_prone_shaped_baoquan_v2 \
  --logger tensorboard \
  --headless

conda run -n env_isaaclab python scripts/rsl_rl/train_t800.py \
  --task_variant getup_supine_shaped \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json \
  --num_envs 1024 \
  --max_iterations 500 \
  --device cuda:0 \
  --run_name t800_getup_supine_shaped_baoquan_v2b \
  --logger tensorboard \
  --headless
```

Playback:

```bash
cd whole_body_tracking

conda run -n env_isaaclab python scripts/rsl_rl/play_t800.py \
  --task_variant getup_prone_shaped \
  --load_run 2026-09-08_01-11-10_t800_getup_prone_shaped_baoquan_v2 \
  --checkpoint model_499.pt \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json \
  --num_envs 1 \
  --device cuda:0 \
  --video \
  --video_length 500 \
  --headless

conda run -n env_isaaclab python scripts/rsl_rl/play_t800.py \
  --task_variant getup_supine_shaped \
  --load_run 2026-09-09_07-53-44_t800_getup_supine_shaped_baoquan_v2b \
  --checkpoint model_499.pt \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json \
  --num_envs 1 \
  --device cuda:0 \
  --video \
  --video_length 500 \
  --headless
```

Evaluation:

```bash
cd whole_body_tracking

conda run -n env_isaaclab python scripts/rsl_rl/evaluate_t800_getup_direct_policy.py \
  --task Getup-Direct-T800-Prone-Shaped-v0 \
  --output /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/training/prone_shaped_v2_eval.json \
  --num_envs 64 \
  --episodes 5 \
  --min_success_rate 0.95 \
  --headless \
  --device cuda:0 \
  --load_run 2026-09-08_01-11-10_t800_getup_prone_shaped_baoquan_v2 \
  --checkpoint model_499.pt \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json

conda run -n env_isaaclab python scripts/rsl_rl/evaluate_t800_getup_direct_policy.py \
  --task Getup-Direct-T800-Supine-Shaped-v0 \
  --output /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/training/supine_shaped_v2b_eval.json \
  --num_envs 64 \
  --episodes 5 \
  --min_success_rate 0.95 \
  --headless \
  --device cuda:0 \
  --load_run 2026-09-09_07-53-44_t800_getup_supine_shaped_baoquan_v2b \
  --checkpoint model_499.pt \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json
```

Current shaped pilot result:

- prone shaped v2: `0/320` success; visual playback reaches a seated/semi-seated posture, not stance.
- supine shaped v2b: `0/320` success; visual playback reaches a seated/semi-seated posture, not stance.

These policies are not real-robot deployment candidates. Continue with a
staged curriculum: prone/supine to seated or kneeling, seated/kneeling to
crouched PD stand, then crouched PD stand to measured boxing guard.
