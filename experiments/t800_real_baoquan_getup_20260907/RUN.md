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

Completed v3.1 checkpoints:

```text
supine: whole_body_tracking/logs/rsl_rl/t800_flat/2026-09-09_08-33-27_t800_getup_supine_staged_baoquan_v31_guard/model_799.pt
prone: whole_body_tracking/logs/rsl_rl/t800_flat/2026-09-09_08-34-24_t800_getup_prone_staged_baoquan_v31_guard/model_799.pt
```

Visual review artifacts:

```text
results/t800_real_baoquan_getup_20260907/training/long_v31_guard/supine_v31_guard_play.mp4
results/t800_real_baoquan_getup_20260907/training/long_v31_guard/prone_v31_guard_play.mp4
results/t800_real_baoquan_getup_20260907/training/long_v31_guard/staged_v31_guard_side_by_side.mp4
results/t800_real_baoquan_getup_20260907/training/long_v31_guard/staged_v31_guard_keyframes.jpg
```

Corrected 320-rollout gate with trajectory diagnostics:

| run | rollouts | final success | any success | max height | min tilt | min height err | min joint err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| supine v3.1 guard | 320 | 0.0 | 0.0 | 1.0780 m | 1.2418 rad | 0.0091 m | 1.0368 rad |
| prone v3.1 guard | 320 | 0.0 | 0.0 | 0.8298 m | 1.4315 rad | 0.0003 m | 1.4508 rad |

Interpretation: v3.1 learned to reach useful root height, but it did not align
the base upright or converge to the measured boxing-guard joint target. The
`getup_guard_stability` term stayed effectively zero in TensorBoard, so it was
too strict to act as dense shaping and mostly behaved like an unreachable final
gate.

## Curriculum Direct-RL v3.2

The v3.2 curriculum variants keep the strict final success gate but add
reachable high-pose shaping once the root is above `0.58` m:

- `getup_prone_curriculum` -> `Getup-Direct-T800-Prone-Curriculum-v0`
- `getup_supine_curriculum` -> `Getup-Direct-T800-Supine-Curriculum-v0`
- `getup_mixed_curriculum` -> `Getup-Direct-T800-Mixed-Curriculum-v0`

New high-pose terms:

- `getup_high_upright`: high-root-height-gated upright reward
- `getup_high_joint_pose`: high-root-height-gated measured baoquan pose reward
- `getup_high_low_velocity`: high-root-height-gated root and joint velocity hold reward

Smoke result on 2026-09-09: both `getup_supine_curriculum` and
`getup_prone_curriculum` constructed and trained for one iteration. The reward
manager exposes `20` terms, including all three high-pose curriculum terms.

Launched tmux long training on 2026-09-09:

```text
tmux session: t800_supine_v32_curriculum
run name: t800_getup_supine_curriculum_baoquan_v32
device: cuda:0
iterations: 1200
log: results/t800_real_baoquan_getup_20260907/training/long_v32_curriculum/supine_v32_curriculum_train.log

tmux session: t800_prone_v32_curriculum
run name: t800_getup_prone_curriculum_baoquan_v32
device: cuda:1
iterations: 1200
log: results/t800_real_baoquan_getup_20260907/training/long_v32_curriculum/prone_v32_curriculum_train.log
```

TensorBoard is available from this machine with:

```bash
conda run --no-capture-output -n env_isaaclab tensorboard \
  --logdir /mnt/data/yangky/test/humanoid_robot_combat/whole_body_tracking/logs/rsl_rl/t800_flat \
  --host 0.0.0.0 \
  --port 6006
```

The current live session is `t800_tensorboard`; stop it with:

```bash
tmux kill-session -t t800_tensorboard
```

v3.2 visual artifacts:

```text
results/t800_real_baoquan_getup_20260907/training/long_v32_curriculum/supine_v32_curriculum_play.mp4
results/t800_real_baoquan_getup_20260907/training/long_v32_curriculum/prone_v32_curriculum_play.mp4
results/t800_real_baoquan_getup_20260907/training/long_v32_curriculum/curriculum_v32_side_by_side.mp4
results/t800_real_baoquan_getup_20260907/training/long_v32_curriculum/curriculum_v32_keyframes.jpg
```

v3.2 continuation and v3.3 guard-focus evaluations both remained at `0/320`
for the full measured baoquan gate. However, the trajectory diagnostics changed
the failure diagnosis: these runs do briefly reach the right height and an
upright base. The remaining blocker is holding that state while reducing the
measured baoquan max-joint error below the final threshold.

## Stand-First Curriculum v3.4

The v3.4 branch fixes a target propagation bug found during the v3.3 analysis:
`apply_getup_target_json` previously updated the direct get-up target stored on
the env config and the original `motion_body_pos/motion_body_ori` rewards, but
new dense reward terms with their own `target_joint_pos` params could still use
`T800_APPROX_BOXING_READY`. The measured and approximate targets differ by up
to `1.54574 rad` per joint, so this could keep the policy near the wrong guard
pose even while `--getup_target_json` was supplied.

The fix updates every observation/reward term containing `target_joint_pos` and
logs the updated term list at startup. v3.4 also adds a stand-first reward
stack:

- `getup_stand_stable`: target height, uprightness, and comfortable root speed
- `getup_near_success`: wide joint-band near-baoquan bonus
- `getup_joint_progress`: max-joint-error progress once high and roughly upright
- `getup_high_root_low_velocity` and `getup_high_joint_low_velocity`: separate
  high/upright-gated hold terms so the product is not always zero

Current live run:

```text
tmux session: t800_v34_loop
active training: t800_supine_v34
run name: t800_getup_supine_curriculum_baoquan_v34_r1
resume source: 2026-09-09_19-15-31_t800_getup_supine_curriculum_baoquan_v33_r5/model_11500.pt
log: results/t800_real_baoquan_getup_20260907/training/long_v34_stand_first/supine_r1_train.log
```

Early v3.4 signal is healthier than v3.2/v3.3 for supine: `anchor_pos=0`,
`getup_stand_stable` is nonzero, and the separate root/joint hold rewards are
nonzero. It still needs the orchestrator's 320-rollout gate before any
deployment discussion.

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
results/t800_real_baoquan_getup_20260907/training/long_v31_guard/supine_v31_guard_eval.json
results/t800_real_baoquan_getup_20260907/training/long_v31_guard/prone_v31_guard_eval.json
results/t800_real_baoquan_getup_20260907/training/long_v31_guard/supine_v31_guard_play.mp4
results/t800_real_baoquan_getup_20260907/training/long_v31_guard/prone_v31_guard_play.mp4
results/t800_real_baoquan_getup_20260907/training/long_v31_guard/staged_v31_guard_side_by_side.mp4
results/t800_real_baoquan_getup_20260907/training/long_v31_guard/staged_v31_guard_keyframes.jpg
results/t800_real_baoquan_getup_20260907/training/long_v32_curriculum/supine_v32_curriculum_play.mp4
results/t800_real_baoquan_getup_20260907/training/long_v32_curriculum/prone_v32_curriculum_play.mp4
results/t800_real_baoquan_getup_20260907/training/long_v32_curriculum/curriculum_v32_side_by_side.mp4
results/t800_real_baoquan_getup_20260907/training/long_v32_curriculum/curriculum_v32_keyframes.jpg
results/t800_real_baoquan_getup_20260907/training/long_v34_stand_first/alignment_check.json
```

## Deployment Decision

Do not deploy these direct get-up policies to the real T800. They have not passed the 320-rollout gate and their ONNX observation contract is the 110-element direct get-up policy input, not the current Native SDK whole-body tracking runtime contract.

Next step: let v3.4 finish, render playback, and run the corrected 320-rollout
gate. Promote a get-up policy only if the rendered motion is physically sane,
the measured target propagation log is correct, and the full gate passes.
