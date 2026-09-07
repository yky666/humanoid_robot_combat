# T800 Direct RL Get-Up

This note covers the reference-free route for training T800 to recover from the
official `pd_stand_x` prone preparation pose and `pd_stand_y` supine preparation
pose directly into a boxing-ready guard.

## Bottom Line

The local codebase now has simulation support for direct RL get-up exploration:

| Task | Reset | Goal |
| --- | --- | --- |
| `Getup-Direct-T800-Prone-v0` | official `pd_stand_x` joint pose, prone root orientation | approximate boxing-ready |
| `Getup-Direct-T800-Supine-v0` | official `pd_stand_y` joint pose, supine root orientation | approximate boxing-ready |
| `Getup-Direct-T800-Mixed-v0` | half prone, half supine | approximate boxing-ready |

This is not yet a hardware-ready path. The target named
`T800_APPROX_BOXING_READY` is only a local engineering placeholder. Before any
real robot deployment, replace it with a measured and approved T800 boxing-idle
joint target, then retrain and requalify.

Pure RL is possible in principle, but it is the harder route. The policy must
discover a long contact-rich sequence from sparse rewards: arm support, leg
scissoring or rolling, foot placement, upright capture, and stable guard. With
no reference motion, expect many more samples and more reward shaping than the
whole-body-tracking route. The recommended competition path remains:

```text
official PD prep pose -> reference-guided recovery PPO -> video review -> 320 rollout gate -> MNN -> isolated hardware graph
```

Use direct RL as a research branch or as a way to improve a reference-guided
policy after a reliable recovery behavior exists.

## Official Context

The public EngineAI `urkl_exams` SDK update adds the two preparation poses:

```text
LB+X -> pd_stand_x, prone recovery preparation
LB+Y -> pd_stand_y, supine recovery preparation
```

Those files are static PD targets, not learned get-up controllers. The public
SDK also exposes `supine_to_stance`, but no public `prone_to_stance` controller
was found in the open repository. The public EngineAI RL Lab currently describes
T800 whole-body tracking tasks; direct RL walking is listed as future input, so
there is no official open direct-RL T800 get-up environment to copy today.

## What Was Added

The direct task reuses the existing IsaacLab/rsl_rl stack and T800 model:

- `t800_mdp.reset_t800_getup_pose`: reset from `pd_stand_x`, `pd_stand_y`, or a
  mixed prone/supine batch.
- `t800_mdp.getup_target_joint_error`: target-pose observation.
- `t800_mdp.getup_root_height_exp`: reward for reaching guard height.
- `t800_mdp.getup_upright_exp`: reward for upright base orientation.
- `t800_mdp.getup_target_joint_pose_exp`: reward for converging to guard joint
  pose.
- `t800_mdp.getup_success_bonus`: sparse terminal-style success reward.
- `t800_mdp.getup_head_contact`: catastrophic head-contact termination.
- `t800_mdp.getup_root_xy_out_of_bounds`: competition-area safety termination.

The direct actor observes target joint error, root height error, projected
gravity, base velocity, policy-ordered joint state, and previous action. It does
not consume a motion trajectory, and its ONNX export emits only `actions`.

## Smoke Training

Run the two orientations separately:

```bash
cd whole_body_tracking

CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/train_t800.py \
  --task_variant getup_supine \
  --num_envs 64 \
  --max_iterations 1 \
  --headless \
  --device cuda:0 \
  --run_name direct_getup_supine_smoke \
  --logger tensorboard

CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/train_t800.py \
  --task_variant getup_prone \
  --num_envs 64 \
  --max_iterations 1 \
  --headless \
  --device cuda:0 \
  --run_name direct_getup_prone_smoke \
  --logger tensorboard
```

After the smoke passes, train each orientation independently:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/train_t800.py \
  --task_variant getup_supine \
  --num_envs 2048 \
  --max_iterations 60000 \
  --headless \
  --device cuda:0 \
  --run_name direct_getup_supine_v1 \
  --logger tensorboard
```

Repeat with `--task_variant getup_prone`. Only train `getup_mixed` after both
single-orientation policies work.

## Render Review

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/play_t800.py \
  --task_variant getup_supine \
  --load_run <run-directory> \
  --checkpoint <model.pt> \
  --num_envs 25 \
  --video \
  --video_length 800 \
  --headless \
  --device cuda:0
```

Reject policies with head contact, violent impacts, impossible sliding,
excessive lateral travel, self-collision, foot tunnelling, or an upright pose
that cannot hold for at least one second.

## 320-Rollout Gate

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/evaluate_t800_getup_direct_policy.py \
  --task Getup-Direct-T800-Supine-v0 \
  --output artifacts/recovery/direct_supine_rollout_report.json \
  --num_envs 64 \
  --episodes 5 \
  --min_success_rate 0.95 \
  --headless \
  --device cuda:0 \
  --load_run <run-directory> \
  --checkpoint <model.pt>
```

The report records the checkpoint SHA-256, final success count, final tilt,
height error, joint error, and root speed. Passing this gate is required but not
sufficient for hardware use.

## Deployment Implication

The current Native SDK deployment runner is built around the whole-body-tracking
contract:

```text
obs:     float32 [1, 140]
actions: float32 [1, 25]
plus reference trajectory outputs
```

Direct get-up policies have a different observation contract and no reference
trajectory. For hardware deployment, add a separate direct-RL runner or an
adapter that reproduces the exact direct observations from IMU, joint state,
base estimate, and the selected target guard pose. Do not put a direct get-up
MNN into the existing tracking runner and expect it to behave correctly.

The safe integration order is:

```text
pd_stand_x -> passive -> direct_getup_prone  -> boxing_ready
pd_stand_y -> passive -> direct_getup_supine -> boxing_ready
any active state -> passive
```

Each hardware package still needs ONNX/MNN parity checks, action clipping,
finite-output checks, torque/speed limits, IMU orientation guards, head-contact
guards where observable, timeout fallback, isolated executor deployment under
`/home/user/projects`, and a rollback path to `robotics.service`.

## Official Sources

- [T800 操作指南](https://ucnj18iantas.feishu.cn/drive/folder/FfrBfrAMxlC37QdFjNacYCeznXd)
- [T800 Operation Guide](https://ucnj18iantas.feishu.cn/drive/folder/RjF5fyXvAl4nwRdyzSvcIWEQnZt)
- [EngineAI developer documentation](https://engineai.com.cn/open/docs)
- [EngineAI Native SDK `urkl_exams`](https://github.com/engineai-robotics/engineai_robotics_native_sdk/commits/urkl_exams)
- [EngineAI RL Lab](https://github.com/engineai-robotics/engineai_rl_lab)
