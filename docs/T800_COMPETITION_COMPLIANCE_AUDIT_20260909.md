# T800 Competition Compliance Audit

Audit date: 2026-09-09 (Asia/Shanghai)

## Rule Assumptions

This audit applies the competition constraints stated by the operator:

- Locomotion must not use EngineAI's official locomotion controller or policy.
- Mimic actions must not directly use EngineAI official action references.

If the written event rules define narrower exceptions, this document should be
updated before packaging a final competition build.

## Standing Combat Actions

The existing standing-action policies have valid simulation evidence under the
repository's formal 320-rollout gate:

| Action | Reference file | Best checkpoint | Formal result | Runtime contract | Competition status |
| --- | --- | --- | ---: | --- | --- |
| Front kick | `kick_push_left_g17_stageii_tracking.npz` | `model_44991.pt` | 320/320 | MNN `obs[1,140] -> actions[1,25]` | Not competition-clean under no-official-mimic rule |
| Straight punch | `punch_cross_left_e3_stageii_tracking.npz` | `model_4999.pt` | 320/320 | MNN `obs[1,140] -> actions[1,25]` | Not competition-clean under no-official-mimic rule |
| Hook punch | `punch_hook_left_e5_stageii_tracking.npz` | `model_33796.pt` | 320/320 | MNN `obs[1,140] -> actions[1,25]` | Not competition-clean under no-official-mimic rule |
| Left jab | `punch_jab_left_e1_stageii_tracking.npz` | `model_4999.pt` | 316/320 | MNN `obs[1,140] -> actions[1,25]` | Not competition-clean under no-official-mimic rule |

Evidence files:

- `results/t800_canonical_v1_20260902/qualification/reports/front_kick_r9_eval.json`
- `results/t800_canonical_v1_20260902/qualification/reports/straight_punch_r1_eval.json`
- `results/t800_canonical_v1_20260902/qualification/reports/hook_punch_r7_eval.json`
- `results/t800_canonical_v1_20260902/qualification/reports/jab_left_r1_eval.json`

The SDK bundle validation passes for these four MNN policies, including joint
order, observation dimension, policy hashes, reference trajectory shape, gains,
action scale, clipping, startup blend, and finite values:

```bash
python engineai_native_sdk_integration/deploy_20260904/overlay/tools/validate_qualifier_bundle.py
```

However, `whole_body_tracking/docs/t800_retargeting_logic_20260902.md` labels
the selected front kick, spinning kick, straight punch, hook punch, and left jab
references as `official mimic`. The canonical manifest also records them with
`official_target` labels. Therefore the current accepted standing actions are
engineering-qualified but should not be used as final competition mimic entries
under the no-official-mimic rule.

They remain useful as:

- internal T800 runtime-contract tests;
- low-amplitude hardware smoke tests when the robot is fully secured;
- baselines for training new competition-clean actions from non-official
  recorded or retargeted references.

## Spinning / Roundhouse Kick

There is no accepted spinning-kick or roundhouse-kick policy for deployment.

- The canonical `spinning_kick` line failed before the formal 320-rollout gate
  because its end-effector termination rate exceeded the configured maximum.
- The independently imported `urkl_roundhouse_540_model33778` policy only
  achieved 47/320 in the corrected IsaacLab formal gate.

Result: do not expose spinning or roundhouse kick in a competition or hardware
deployment graph until a new policy passes the full gate.

## Recovery

The archived `recovery_supine` result uses EngineAI's public
`T800_supine_to_stance.mnn`. It is not a custom policy and is not competition
clean if official recovery controllers are disallowed.

The direct RL prone/supine get-up line is still in progress. The v34
stand-first run fixed measured baoquan-target propagation across reward terms,
but it had not yet produced a passing 320-rollout report at the time of this
audit.

## Locomotion

The current real-robot deployment graph still maps `walk` to EngineAI's official
Native SDK runner:

```yaml
motion: walk
runner:
  - name: rl_walking_example_runner
    param_tag: rl_walking_example
```

This is not competition compliant under the no-official-locomotion rule. The
keyboard `w` and gamepad `RB+X` walking controls are therefore debug-only until
they are replaced by a custom locomotion policy and runner.

Available local data:

- `/mnt/data/yangky/test/datasets/urkl_locomotion_260901/gmr_pkl_all`
- `/mnt/data/yangky/test/datasets/urkl_locomotion_260901/tracking_npz`
- directions: forward, backward, left, right

Existing code support:

- There is a conversion queue,
  `whole_body_tracking/scripts/t800_locomotion_pkl_to_npz_queue.sh`.
- There is no dedicated T800 velocity-command locomotion environment registered
  in `train_t800.py` or `play_t800.py`.
- There is no AMP discriminator or AMP-style training loop in the active
  `whole_body_tracking` task tree.
- The existing `rl_dance_example_runner` is time-indexed reference tracking; it
  is not a commandable joystick locomotion runner.

Result: custom locomotion is not trained and is not directly deployable today.

## Recommended Competition-Clean Locomotion Route

Train a new command-conditioned T800 locomotion policy that starts from and
returns to the measured baoquan guard:

1. Add a T800 locomotion environment with commands
   `[v_x, v_y, yaw_rate]`, sampled over conservative ranges first.
2. Use measured `baoquan` joint targets as the upper-body guard prior. Reward
   arms, torso, and head for staying near guard while legs track commanded base
   velocity.
3. Use only allowed data for style regularization. The local forward/back/left/
   right recordings can seed phase/style rewards. Do not use official locomotion
   clips or official walking policies.
4. AMP is feasible, but requires adding an AMP discriminator/training loop. A
   lighter first pass is PPO with velocity tracking, foot-contact regularizers,
   upper-body guard rewards, smoothness, joint-limit, velocity-limit, and torque
   margins.
5. Evaluate with fixed command grids and randomized pushes:
   - zero command hold in baoquan;
   - forward/back/left/right;
   - yaw turns;
   - mixed diagonal commands;
   - transition back to baoquan;
   - 320-rollout formal gate before hardware.
6. Add a new Native SDK runner or extend the custom runner for velocity-command
   input. The current time-indexed action runner cannot provide joystick-like
   continuous locomotion control.

## Deployment Decision

Current deployability:

- Hardware/runtime deployable for controlled internal tests:
  front kick, straight punch, hook punch, left jab.
- Not competition deployable under the stated rule:
  official walking, official supine recovery, and official-mimic-derived action
  references.
- Not deployable:
  spinning/roundhouse kick, custom prone/supine RL get-up, custom locomotion.

The next competition-safe implementation target should be custom baoquan
locomotion. After that, retrain combat actions from non-official references with
`baoquan -> action -> baoquan` entry/exit segments so the policy and real-robot
state machine share the same initial and terminal posture.
