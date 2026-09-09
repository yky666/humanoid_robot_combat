# T800 Real Baoquan Get-Up Run

Date: 2026-09-09

## Target

The terminal boxing-guard target comes from the approved real T800 `pdstand2baoquan` log render. The stable hold action is:

```text
results/t800_real_baoquan_getup_20260907/baoquan_tail_reference_2s.npz
```

## Direct-RL v1

Commands:

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

Result: failed the corrected 320-rollout gate.

| Run | Trials | Success Rate | Final Tilt | Height Error | Joint Error |
| --- | ---: | ---: | ---: | ---: | ---: |
| supine v1 | 320 | 0.0 | 3.0801 rad | 0.4695 m | 2.6550 rad |
| prone v1 | 320 | 0.0 | 3.0781 rad | 0.4698 m | 3.7007 rad |

## Shaped Direct-RL v2

The shaped task adds broad upright and height progress rewards and removes head-contact early termination during cold-start training. Head contact remains penalized.

Commands:

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

Training end metrics:

| Run | Mean Reward | Mean Episode Length | Upright Progress | Height Progress | Success Bonus |
| --- | ---: | ---: | ---: | ---: | ---: |
| prone shaped v2 | 50.4721 | 500.0 | 2.8253 | 0.4481 | 0.0 |
| supine shaped v2b | 49.6972 | 500.0 | 2.8349 | 0.4551 | 0.0 |

Corrected 320-rollout gate:

| Run | Trials | Success Rate | Final Tilt | Height Error | Joint Error |
| --- | ---: | ---: | ---: | ---: | ---: |
| prone shaped v2 | 320 | 0.0 | 3.1117 rad | 0.4702 m | 3.7007 rad |
| supine shaped v2b | 320 | 0.0 | 3.1110 rad | 0.4697 m | 2.6558 rad |

Visual result: both shaped policies learn a sit-up / semi-seated posture, not a full standing get-up.

Review artifacts:

```text
results/t800_real_baoquan_getup_20260907/training/shaped_v2_side_by_side.mp4
results/t800_real_baoquan_getup_20260907/training/shaped_v2_keyframes.jpg
```

## Staged Direct-RL v3

The next direct-RL variant keeps the v2 dense upright/height rewards and adds:

- staged root-height rewards at `0.35`, `0.45`, `0.55`, `0.65`, and `0.72` m
- height-gated stability reward once the root is above `0.50` m
- boxing-guard stability reward gated by target height, upright tilt, and terminal joint-pose error
- joint-angle margin penalty outside the inner `90%` of the soft joint range
- joint-velocity and applied-torque penalties above `80%` of configured limits

Limit source:

```text
GMR/assets/t800/serial_t800.urdf
engineai_robotics_native_sdk/assets/resource/robot/t800/urdf/serial_t800.urdf
whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/t800.py
```

The IsaacLab robot config already uses `soft_joint_pos_limit_factor=0.9`; v3 adds the additional policy-level margin above so the learned get-up leaves deployable headroom.

Smoke command:

```bash
cd whole_body_tracking
conda run -n env_isaaclab python scripts/rsl_rl/train_t800.py \
  --task_variant getup_supine_staged \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json \
  --num_envs 64 \
  --max_iterations 1 \
  --device cuda:0 \
  --run_name t800_getup_supine_staged_smoke \
  --logger tensorboard \
  --headless
```

Smoke result on 2026-09-09: both `getup_supine_staged` and
`getup_prone_staged` constructed successfully. The actor observation shape is
`110`, action shape is `25`, and the reward manager exposes all staged terms:
`getup_height_stages`, `getup_stability`, `joint_margin`,
`joint_velocity_margin`, and `joint_torque_margin`.

Follow-up smoke on 2026-09-09 added `getup_guard_stability`; the staged task
constructs successfully with `17` reward terms. This v3.1 setting is the one to
use for the next long training run.

Longer training, after smoke passes:

```bash
cd whole_body_tracking
conda run -n env_isaaclab python scripts/rsl_rl/train_t800.py \
  --task_variant getup_prone_staged \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json \
  --num_envs 1024 \
  --max_iterations 800 \
  --device cuda:1 \
  --run_name t800_getup_prone_staged_baoquan_v3 \
  --logger tensorboard \
  --headless

conda run -n env_isaaclab python scripts/rsl_rl/train_t800.py \
  --task_variant getup_supine_staged \
  --getup_target_json /mnt/data/yangky/test/humanoid_robot_combat/results/t800_real_baoquan_getup_20260907/measured_boxing_ready.json \
  --num_envs 1024 \
  --max_iterations 800 \
  --device cuda:0 \
  --run_name t800_getup_supine_staged_baoquan_v3 \
  --logger tensorboard \
  --headless
```

Launched tmux long training on 2026-09-09 with the v3.1 guard-stability reward:

```text
tmux session: t800_supine_v31_guard
run name: t800_getup_supine_staged_baoquan_v31_guard
device: cuda:0
log: results/t800_real_baoquan_getup_20260907/training/long_v31_guard/supine_v31_guard_train.log

tmux session: t800_prone_v31_guard
run name: t800_getup_prone_staged_baoquan_v31_guard
device: cuda:1
log: results/t800_real_baoquan_getup_20260907/training/long_v31_guard/prone_v31_guard_train.log
```

Startup note: launching both IsaacLab jobs at the exact same second caused the
prone run to fail during temporary URDF/USD conversion. The active long runs
were relaunched with live `conda run --no-capture-output` logs and staggered
startup.

## Artifacts

```text
results/t800_real_baoquan_getup_20260907/training/prone_shaped_v2_model_499.pt
results/t800_real_baoquan_getup_20260907/training/prone_shaped_v2_policy.onnx
results/t800_real_baoquan_getup_20260907/training/prone_shaped_v2_play.mp4
results/t800_real_baoquan_getup_20260907/training/prone_shaped_v2_eval.json
results/t800_real_baoquan_getup_20260907/training/shaped_v2_side_by_side.mp4
results/t800_real_baoquan_getup_20260907/training/shaped_v2_keyframes.jpg
results/t800_real_baoquan_getup_20260907/training/supine_shaped_v2b_model_499.pt
results/t800_real_baoquan_getup_20260907/training/supine_shaped_v2b_policy.onnx
results/t800_real_baoquan_getup_20260907/training/supine_shaped_v2b_play.mp4
results/t800_real_baoquan_getup_20260907/training/supine_shaped_v2b_eval.json
```

## Deployment Decision

Do not deploy these direct get-up policies to the real T800. They have not passed the 320-rollout gate and their ONNX observation contract is the 110-element direct get-up policy input, not the current Native SDK whole-body tracking runtime contract.

Next step: train a staged curriculum with an intermediate seated/kneeling posture, then crouched PD stand, then the measured boxing guard.
