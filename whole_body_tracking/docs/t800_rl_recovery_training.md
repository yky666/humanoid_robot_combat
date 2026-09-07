# T800 RL Recovery Training and Deployment

This document defines the reproducible path from the official prone/supine PD
preparation poses to an upright boxing-ready state. It deliberately separates
available infrastructure from policies that have actually passed qualification.

## Audit Result

The repository can support recovery training today, but the two requested
policies have not yet been trained or qualified.

| Capability | Status | Evidence or gap |
| --- | --- | --- |
| T800 reference-motion PPO | Available | IsaacLab tracking task, adaptive phase sampling, contact sensor, residual joint actions |
| Exact motion-phase reset | Available | The command manager writes root, joint, and velocity state from any reference frame |
| Recovery-specific contacts | Added | `Tracking-Flat-T800-Recovery-v0` allows expected body/limb contact and penalizes head contact |
| Official `pd_stand_x/y` joint seeds | Available | Integrated `pose_x.yaml` and `pose_y.yaml` from EngineAI commit `0d75937` |
| Public official supine policy | Available but quarantined for hardware | The Native SDK publishes `supine_to_stance`; earlier custom-runner hardware output was unsafe |
| Public official prone policy | Not found | The open SDK publishes `pd_stand_x`, not a prone get-up controller |
| Public official boxing-ready pose | Not found | Product demonstrations are not a joint-space/runtime contract |
| Independent prone/supine RL policies | Not trained | New reference, training, video, and 320-rollout evidence are required |

The existing environment is reference-guided residual RL, not a sparse-reward
controller that discovers a get-up from only an upright goal. It is the shortest
deployment-compatible route because it retains the existing runtime contract:

```text
observations: float32 [1, 140]
actions:      float32 [1, 25]
```

A reference-free, goal-conditioned recovery task is possible, but it would need
a new observation/reward definition and a corresponding Native SDK runner. It
should be treated as a separate research branch, not mixed into the current
deployment path.

## Required Reference Assets

Build and review two independent 50 Hz tracking motions:

```text
prone:  pd_stand_x hold -> physically valid prone get-up -> boxing-ready hold
supine: pd_stand_y hold -> physically valid supine get-up -> boxing-ready hold
```

Each motion needs the full tracking NPZ contract: `fps`, `joint_names`,
`joint_order_version`, `joint_pos`, `joint_vel`, `body_names`, `body_pos_w`,
`body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`. The first body must be
`LINK_BASE`.

Use a measured T800 boxing-ready hold when an official controller exposes the
desired guard. Otherwise, the local martial-arts
`transition_male2_crouch_to_ready` asset is only a candidate and must pass visual
review, collision checking, balance simulation, and joint-limit checks before it
becomes the common endpoint. Do not infer joint targets from a video.

For recovery motion sources, prefer this order:

1. Competition-supported T800 telemetry captured from the official controller.
2. A physically simulated T800 recovery authored from measured preparation and guard poses.
3. Retargeted human get-up motion followed by physics-aware cleanup.

Reversing a fall clip can provide an initialization candidate, but it is not
evidence that the reversed motion is dynamically feasible. Arm-ground strikes,
rollovers, and leg-scissor phases must be represented with their real contacts.

The nominal first frame must match the official PD target. The runtime must also
check IMU orientation, body height, contact state, and the measured pose after
the short `passive` settling interval. Joint-angle agreement alone cannot
distinguish an upright robot from a robot on the floor.

## Validate References

Keep one separately approved `boxing_ready_hold_tracking.npz` as the shared
terminal target. Validate each candidate before training:

```bash
cd whole_body_tracking

python scripts/t800_validate_recovery_reference.py \
  artifacts/recovery/prone_pd_x_to_boxing_ready_tracking.npz \
  --orientation prone \
  --ready-reference artifacts/recovery/boxing_ready_hold_tracking.npz \
  --output artifacts/recovery/prone_reference_report.json

python scripts/t800_validate_recovery_reference.py \
  artifacts/recovery/supine_pd_y_to_boxing_ready_tracking.npz \
  --orientation supine \
  --ready-reference artifacts/recovery/boxing_ready_hold_tracking.npz \
  --output artifacts/recovery/supine_reference_report.json
```

The default offline limits are 0.08 rad for the initial PD match, 0.08 rad for
the terminal guard match, 0.05 rad for hold drift, and 3.5 m planar displacement.
The 3.5 m value is an engineering margin inside the competition's 8 m diameter
recovery area, not a replacement for full simulation.

## Train and Render

Run prone and supine as separate policies. Start with one iteration, inspect the
termination metrics, then launch the full run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/train_t800.py \
  --task_variant recovery \
  --motion_file artifacts/recovery/prone_pd_x_to_boxing_ready_tracking.npz \
  --num_envs 64 \
  --max_iterations 1 \
  --headless \
  --device cuda:0 \
  --run_name recovery_prone_smoke \
  --logger tensorboard

CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/train_t800.py \
  --task_variant recovery \
  --motion_file artifacts/recovery/prone_pd_x_to_boxing_ready_tracking.npz \
  --num_envs 1024 \
  --max_iterations 30000 \
  --headless \
  --device cuda:0 \
  --run_name recovery_prone_v1 \
  --logger tensorboard
```

Repeat with the supine reference. Render the selected checkpoint from phase zero:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/play_t800.py \
  --task_variant recovery \
  --load_run <run-directory> \
  --checkpoint <model.pt> \
  --motion_file artifacts/recovery/prone_pd_x_to_boxing_ready_tracking.npz \
  --num_envs 25 \
  --video \
  --video_length 800 \
  --headless \
  --device cuda:0
```

Review at least front, side, and elevated views. Reject head contact, foot or
knee tunnelling, self-collision, implausible sliding, violent terminal impact,
or a terminal stance that cannot hold for one second.

## Formal Qualification

The minimum automated gate remains 64 environments times 5 fresh batches:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/evaluate_t800_policy.py \
  --task Tracking-Flat-T800-Recovery-v0 \
  --motion_file artifacts/recovery/prone_pd_x_to_boxing_ready_tracking.npz \
  --output artifacts/recovery/prone_rollout_report.json \
  --num_envs 64 \
  --episodes 5 \
  --min_success_rate 0.95 \
  --load_run <run-directory> \
  --checkpoint <model.pt> \
  --headless
```

Passing requires all of the following, not only a high average reward:

- at least 304/320 successful rollouts;
- no NaN/Inf output and no joint-limit violation;
- no head impact and bounded intended body contacts;
- terminal upright and boxing-ready hold for at least one second;
- acceptable torque, speed, slip, and base-orientation traces;
- the exact prone or supine entry distribution, including passive settling;
- displacement within the designated recovery area;
- human review of the rendered motion.

The archived 320/320 public supine report used EngineAI's Native SDK policy and
the vendor lie-down trajectory. It does not qualify either new
`pd_stand_x/y -> boxing_ready` policy.

## ONNX to MNN and Staging

Playback exports the training policy to `exported/policy.onnx`. Extract and
check the actor-only graph, then convert it with the same MNN 2.9.5 toolchain as
the ARM64 runtime:

```bash
python ../engineai_native_sdk_integration/deploy_20260904/overlay/tools/prepare_qualifier_policy.py \
  <run-directory>/exported/policy.onnx \
  <run-directory>/exported/recovery_prone_actor.onnx

MNNConvert \
  -f ONNX \
  --modelFile <run-directory>/exported/recovery_prone_actor.onnx \
  --MNNModel <run-directory>/exported/recovery_prone.mnn \
  --bizCode t800_recovery
```

Compare ONNX and MNN outputs over recorded observations before packaging. Then
stage only artifacts whose reference and rollout reports passed:

```bash
python scripts/t800_stage_recovery_deployment.py \
  --orientation prone \
  --policy-onnx <run-directory>/exported/recovery_prone_actor.onnx \
  --policy-mnn <run-directory>/exported/recovery_prone.mnn \
  --trajectory artifacts/recovery/prone_pd_x_to_boxing_ready_tracking.npz \
  --reference-report artifacts/recovery/prone_reference_report.json \
  --rollout-report artifacts/recovery/prone_rollout_report.json \
  --output-root ../engineai_native_sdk_integration/generated \
  --tag rl_recovery_v1
```

The evaluator records the checkpoint SHA-256 in its report, and playback embeds
the same checkpoint hash in ONNX metadata. The stager requires those hashes to
match, copies the policy and trajectory, writes a Native SDK configuration, and
records the package hashes in `DEPLOYMENT_MANIFEST.json`. Its status remains
`staged_not_hardware_approved`.

## Real-Robot FSM

First integrate the packages into a recovery-only graph:

```text
pd_stand_x -> passive -> qualifier_recovery_prone  -> boxing_ready
pd_stand_y -> passive -> qualifier_recovery_supine -> boxing_ready
any active state --LB+RB--> passive
```

The transition from `passive` must select a recovery policy using the retained
orientation/pose latch from `pd_stand_x` or `pd_stand_y`; arbitrary passive
entry must not guess an orientation. Require matching IMU, base height, joint
error, and contact guards before enabling torque. On completion, require a
stable upright hold before entering boxing-ready. A timeout, non-finite output,
joint limit, excessive torque, or orientation mismatch returns to `passive`.

Keep official walking, official supine recovery, existing combat actions, and
the new recovery policies as distinct states and assets. Do not overwrite
`/apps/engineai_robotics`. Copy the qualified package into the independent
project directory, rebuild for ARM64, verify its manifest, stop
`robotics.service`, confirm that no other `src_executor` exists, and test one
orientation at a time under a rated harness over mats.

Only after both recovery-only tests pass may the two states be added to the full
walking and combat graph. Hardware telemetry and the exact deployed hashes must
be added to the qualification archive and pushed with the policies through Git
LFS.

## Official Sources

- [EngineAI RL Lab](https://github.com/engineai-robotics/engineai_rl_lab)
- [EngineAI Native SDK `urkl_exams` update](https://github.com/engineai-robotics/engineai_robotics_native_sdk/commit/0d759376cba552b480f267042d5d069ad5d96b50)
- [EngineAI Native SDK](https://github.com/engineai-robotics/engineai_robotics_native_sdk)
- [EngineAI developer documentation](https://engineai.com.cn/open/docs)
- [URKL online selection resources](https://en.engineai.com.cn/online-selection.html)
- [URKL competition rules](https://www.engineai.com.cn/tournament-rule-detailed.html)
