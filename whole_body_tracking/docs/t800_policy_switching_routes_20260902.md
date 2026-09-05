# T800 Policy Switching Routes, 2026-09-02

This note summarizes the current training state and the two deployment routes for URKL qualifier-style motion switching on EngineAI T800.

## Current Training State

The manually approved qualifier references have been trained as one 6-motion joint policy, not as six complete independent single-motion policies.

Approved references:

| Motion | Role | Tracking NPZ | Frames | Individual policy status |
| --- | --- | --- | ---: | --- |
| `kick_push_left_g17_stageii` | front push kick | `artifacts/t800_approved_qualifier_20260902/tracking_npz/kick_push_left_g17_stageii_tracking.npz` | 226 | not found |
| `kick_reverse_spin_cresent_right_g20_stageii` | spinning kick | `artifacts/t800_approved_qualifier_20260902/tracking_npz/kick_reverse_spin_cresent_right_g20_stageii_tracking.npz` | 141 | not found |
| `punch_cross_left_e3_stageii` | straight punch | `artifacts/t800_approved_qualifier_20260902/tracking_npz/punch_cross_left_e3_stageii_tracking.npz` | 162 | not found |
| `punch_hook_left_e5_stageii` | hook punch | `artifacts/t800_approved_qualifier_20260902/tracking_npz/punch_hook_left_e5_stageii_tracking.npz` | 135 | only smoke/no full run found |
| `punch_jab_left_e1_stageii` | straight punch variant | `artifacts/t800_approved_qualifier_20260902/tracking_npz/punch_jab_left_e1_stageii_tracking.npz` | 165 | historical single run exists |
| `recovery_supine_male2_reverse_fall_to_ready` | supine recovery | `artifacts/t800_approved_qualifier_20260902/tracking_npz/recovery_supine_male2_reverse_fall_to_ready_tracking.npz` | 462 | not found |

Current best joint policy:

- Run directory: `/data2/yangky/test/whole_body_tracking/logs/rsl_rl/t800_flat/2026-09-02_12-44-27_approved_qualifier_6_t800_joint_v1_from31000`
- Final checkpoint: `model_38999.pt`
- Exported policy: `exported/policy.onnx`
- Final rollout video: `videos/play/rl-video-step-0.mp4`
- Switch-rollout videos:
  - `videos/qualifier_6_interim_policy_switch/rl-video-step-0.mp4`
  - `videos/qualifier_6_interim_policy_switch_nosnap/rl-video-step-0.mp4`

For final mode switching, the joint policy is the more useful artifact than separate per-action fine-tunes, because one policy can track different references selected by the task state. Separate single-motion policies are still useful for debugging a weak motion, but switching separate low-level policies at runtime adds transition risk.

## Sonic Route

Sonic/GR00T-WholeBodyControl is useful as a planner/mode-switching reference, but the released stack is G1-coupled:

- G1 action dimension is 29, while our T800 policy controls 25 joints.
- G1 planner qpos dimension is `7 + 29 = 36`, while T800 tracking qpos is `7 + 25 = 32`.
- Sonic deploy expects a decoder contract like `obs_dict -> action`; our exported T800 policy uses `obs + time_step -> actions` and additionally exports reference arrays.
- The Sonic planner currently emits G1-layout qpos, so replacing only the low-level decoder is not enough.

Current Sonic/T800 status:

- T800 MuJoCo visual scene exists under `/data2/yangky/test/GR00T-WholeBodyControl`.
- Sonic-style T800 reference replay video exists at `/data2/yangky/test/GR00T-WholeBodyControl/outputs/sonic_t800_approved_qualifier_20260902/t800_approved_qualifier_mode_switch_h264.mp4`.
- This validates labeled reference/mode playback with T800 mesh, but it is not yet a true T800 Sonic planner + encoder + decoder runtime.

To make Sonic production-ready for T800, we would need to train or port:

- a T800 planner that outputs 32-D qpos;
- a T800 encoder/decoder observation contract;
- T800 PD/default pose/action scale/joint maps;
- a runtime adapter for our T800 ONNX or a Sonic-compatible exported wrapper.

This is a higher-R&D route. It may become valuable for richer boxing behavior composition later, but it is not the shortest competition path.

## EngineAI SDK Route

This is the recommended route for the qualifier control loop because it matches the official SDK style:

- MuJoCo simulation is already integrated in `engineai_robotics_native_sdk`.
- Motion/task switching is already represented as SDK task states.
- Virtual gamepad and keyboard-style switching are already supported by the SDK tooling.
- The same FSM-style layout is closer to later robot deployment than the Sonic G1 runtime.

Staged files:

- Mode config: `/data2/yangky/test/engineai_robotics_native_sdk/assets/config/t800/mode_qualifier_approved.yaml`
- Task config: `/data2/yangky/test/engineai_robotics_native_sdk/assets/config/t800/task_motion/qualifier_approved.yaml`
- Per-motion runner configs: `/data2/yangky/test/engineai_robotics_native_sdk/assets/config/t800/rl_qualifier_approved_20260902/qualifier_*.yaml`
- Copied ONNX policy for SDK staging: `/data2/yangky/test/engineai_robotics_native_sdk/assets/config/t800/rl_qualifier_approved_20260902/policies/t800_qualifier_joint_policy.onnx`
- Session launcher: `/data2/yangky/test/engineai_robotics_native_sdk/scripts/run_mujoco_t800_qualifier.sh`
- Tmux session launcher: `/data2/yangky/test/engineai_robotics_native_sdk/scripts/start_t800_qualifier_mujoco_tmux.sh`
- Keyboard-only LCM sender: `/data2/yangky/test/engineai_robotics_native_sdk/scripts/send_t800_qualifier_key.sh`
- Virtual gamepad venv setup: `/data2/yangky/test/engineai_robotics_native_sdk/scripts/setup_virtual_gamepad_venv.sh`

Current key map:

| State | Meaning | Key |
| --- | --- | --- |
| `pd_stand` | stand | `LB + A` |
| `walk` | walking policy | `LB + B` |
| `dance` | default dance slot | `RB + B` |
| `qualifier_front_kick` | front push kick | `RB + A` |
| `qualifier_spinning_kick` | spinning kick | `RB + X` |
| `qualifier_straight_punch` | cross/straight punch | `RB + Y` |
| `qualifier_hook_punch` | hook punch | `LB + X` |
| `qualifier_jab_left` | jab/straight variant | `LB + Y` |
| `qualifier_recovery_supine` | supine recovery | `BACK + A` |
| `passive` | emergency/passive | `LB + RB` |

Keyboard equivalents in `tools/virtual_gamepad/virtual_gamepad.py`:

| Gamepad key | Keyboard |
| --- | --- |
| `LB` | `q` |
| `RB` | `e` |
| `A` | `j` |
| `B` | `k` |
| `X` | `u` |
| `Y` | `i` |
| `BACK` | `F1` |
| `START` | `F2` |

Keyboard-only command examples:

```bash
cd /mnt/data/yangky/test/engineai_robotics_native_sdk
./scripts/send_t800_qualifier_key.sh pd_stand
./scripts/send_t800_qualifier_key.sh front_kick
./scripts/send_t800_qualifier_key.sh spinning_kick
./scripts/send_t800_qualifier_key.sh straight_punch
./scripts/send_t800_qualifier_key.sh hook_punch
./scripts/send_t800_qualifier_key.sh jab_left
./scripts/send_t800_qualifier_key.sh recovery_supine
./scripts/send_t800_qualifier_key.sh passive
```

Important interface gap:

- The SDK `rl_dance_example_runner` was extended to keep normal `.mnn` inference and also load `.onnx` policies through ONNXRuntime.
- Our current exported T800 policy is ONNX and has inputs `obs [1, 140]` and `time_step [1, 1]`, with action output `actions [1, 25]`.
- The qualifier YAML observation list now matches the 140-D training-side policy observation: `command`, `motion_anchor_pos_b`, `motion_anchor_ori_b`, `base_lin_vel`, `base_ang_vel`, `joint_pos`, `joint_vel`, `actions`.
- `/opt/onnxruntime` is present on this machine and sys01, so the ONNX path is practical if the SDK third-party package is installed.

Joint-name boundary:

- Whole-body-tracking/GMR uses a DFS-style T800 list with right-arm/head names `J20..J24` and `J27/J28`.
- EngineAI SDK's current T800 MuJoCo XML uses a contiguous deployed list with right-arm/head names `J18..J22` and `J23/J24`.
- The 25 columns are still in the same semantic order: legs, torso, left arm, right arm, head.
- Therefore the SDK staging configs intentionally keep SDK XML names while consuming the same 25-column trajectory order. Do not directly paste GMR joint-name strings into SDK configs unless the SDK XML is changed at the same time.

Validated sys01 MuJoCo launch on 2026-09-02 evening:

- `t800_qualifier_mujoco` tmux starts MuJoCo, executor, and virtual gamepad windows.
- sys01 is built with `BUILD_ROS2=OFF`, so qualifier sim uses `task_resident/sim_no_ros.yaml`.
- `idle` now auto-transitions to `pd_stand`, preventing a no-control passive drop at startup.
- `pd_stand`, `walk`, and `dance` can manually transition into every qualifier action.
- CLI test succeeded: `pd_stand -> qualifier_front_kick -> pd_stand`.
- Executor log confirmed ONNX load: `obs_dim=140`, `actions=25`, `frames=226` for `qualifier_front_kick`.

Run MuJoCo with the qualifier mode config and virtual gamepad:

```bash
cd /mnt/data/yangky/test/engineai_robotics_native_sdk
./scripts/start_t800_qualifier_mujoco_tmux.sh
```

The default `mode.yaml` is still the normal SDK mode file. The qualifier launchers copy `mode_qualifier_approved.yaml` over `mode.yaml` and keep a timestamped backup next to it.

## How Switching Works

EngineAI SDK switching is an FSM-level state change. The config chain is:

- `mode.yaml` chooses the active config scopes for `sim` or `robot`.
- `task_motion/*.yaml` defines each motion state, its runner, its `param_tag`, allowed transitions, and key combo.
- The input command arbiter listens to the physical or virtual gamepad and asks the FSM to enter the keyed state.
- The selected runner loads parameters by `param_tag`; for example `qualifier_front_kick` maps to `rl_qualifier_approved_20260902/qualifier_front_kick.yaml`.

In the qualifier setup, the native SDK states remain available:

- `pd_stand` uses `pd_stand_runner`.
- `walk` uses `rl_walking_example_runner` and the native walking MNN policy.
- `dance` uses `rl_dance_example_runner` and the native dance MNN policy.

The added qualifier states also use `rl_dance_example_runner`, but with different `param_tag` values. Each qualifier YAML points to the same joint policy ONNX and a different reference trajectory NPZ. So switching from jab to hook/front kick changes the task state's reference trajectory and phase source, while the low-level neural network weights stay the same.

The current runner clamps `policy_step` at the final reference frame, so a selected qualifier action holds its final pose after playback. It does not loop automatically. For interactive review, switch back to `pd_stand`, `walk`, or `passive`, or select another qualifier action from the gamepad.

Joint policy vs single-task policy:

- A joint 6-motion policy is one network trained on multiple approved references. Runtime switching keeps one policy loaded and changes the reference/motion config. This is usually smoother and closer to the final "mode switch" requirement.
- A single-task policy is one network per action. It can overfit a specific motion and is useful for debugging weak actions, but switching means changing policy files or runner configs, which can create discontinuities and extra transition risk.
- For the qualifier, the joint policy should be the main deployment artifact. Single-task fine-tunes are best used as diagnostics or as backups for motions that the joint policy cannot stabilize.

## Dependency State

The missing official binary dependencies were recovered from the official `ghcr.io/engineai-robotics/engineai_robotics_env:latest` image:

- `/opt/engineai_robotics_third_party`
- `/opt/engineai_robotics_hardware`
- Boost 1.74 and urdfdom 3.0 runtime libraries copied into `/opt/engineai_robotics_third_party/lib`

The SDK build scripts now prefer the official third-party CMake packages, including MuJoCo 3.2.3, glog 0.8.0, lcm, yaml-cpp, fmt, Eigen3, pinocchio, and qpOASES. On hosts without ROS Humble, `build.sh` falls back to `BUILD_ROS2=OFF`; this is enough for MuJoCo/FSM/RL-runner validation.

Current dependency status:

- Local machine and sys01 both have `/opt/engineai_robotics_third_party` and `/opt/engineai_robotics_hardware`.
- sys01 `./scripts/build_mujoco.sh` and `./build.sh -j 24` have passed with `BUILD_ROS2=OFF`.
- Local `./scripts/build_mujoco.sh` and `./build.sh -j 24` have passed.
- sys01 virtual gamepad dependencies are installed through `python3-lcm`, `python3-pyqt6`, and `.venv_gamepad`.

The 2026-09-02 19:38 collapse was caused by launch infrastructure, not by a completed policy execution:

- `src_executor` crashed because `task_resident/sim.yaml` requested `ros2_bridge_runner`, while sys01 was built without ROS2.
- `virtual_gamepad.py` also failed because `PyQt6` was missing.
- MuJoCo kept running under gravity with no valid controller command path, so the robot fell.
- The fixed qualifier sim now uses `task_resident/sim_no_ros.yaml` and starts in `pd_stand`.

## Recommendation

Prioritize the EngineAI SDK route for competition readiness:

- It already has the control abstraction we need: stand, walk, action states, and gamepad/keyboard transitions.
- It is closer to the official suggested path and later robot deployment.
- It can use the 6-motion joint policy as a single low-level tracking policy while switching reference tasks.

Keep Sonic as the research branch:

- Use it to study higher-level boxing mode/planner design.
- Porting Sonic properly requires T800 planner + encoder/decoder work, not just inserting our checkpoint.
- Sonic is useful after the SDK route proves the T800 joint policy can switch among required motions in MuJoCo without falling.
