# Roundhouse 540 midpush model33778 package

Source training run:

`/root/gpufree-data/human_robot/whole_body_tracking_engineai/logs/rsl_rl/t800_flat/2026-05-23_01-03-47_normfix_curriculum_540_midpush_k5_u085_a0001_pos125_ori115_vel08_curriculum_scaffold_consolidate2_from_m31779_8192env_2000iter`

Source checkpoint:

`model_33778.pt`

Source motion:

`/root/gpufree-data/human_robot/whole_body_tracking_engineai/data/server_verify_20260513_trainfoot_negative/540huixuantitui_001_50hz_trainfoot_global_c-0048_from_portable.npz`

Task:

`Tracking-Flat-T800-v0`

Package files:

- `checkpoint/model_33778.pt`
- `exported/policy.onnx`
- `exported/deploy_config.yaml`
- `git/human_robot.diff`
- `motion/540huixuantitui_001_50hz_trainfoot_global_c-0048_from_portable.npz`
- `params/agent.yaml`
- `params/env.yaml`
- `params/agent.pkl`
- `params/env.pkl`
- `videos/play/rl-video-step-0.mp4`

Notes:

- The experimental sim2sim package keeps the exported policy in ONNX format.
- `config/sim2sim_540huixuanti.yaml` points to this package for local MuJoCo policy validation.
- The motion file follows the training run's `params/env.yaml` contract.
