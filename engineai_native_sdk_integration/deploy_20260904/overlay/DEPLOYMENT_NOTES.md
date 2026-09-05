# T800 Qualifier Deployment

## Identity

- Package directory: `/home/user/projects/engineai_robotics_qualifier_20260905_recovery`
- SDK upstream commit: `335c60e88772c26c7852d0abd6b3c7439037dd8f`
- Target: T800 Nezha controller, ARM64, ROS 2 Humble
- Official package: `/apps/engineai_robotics` (not modified)
- Verified IMU firmware: `V01.02.06b` (`engineai-imu-update` package `1.0.3`)

## Policy Interface

- Runtime models: one accepted per-motion MNN under `assets/config/t800/rl_qualifier_deploy_20260904/policies/`
- Input: `obs`, float32, shape `[1, 140]`
- Output: `actions`, float32, shape `[1, 25]`
- MNN version used for conversion and validation: 2.9.5

The observation order is command, aligned motion anchor position and
orientation, base linear and angular velocity, joint position, joint velocity,
and previous action. The runner rejects invalid dimensions and non-finite data.

## Motions

- `pd_stand_x` (official prone recovery preparation pose)
- `pd_stand_y` (official supine recovery preparation pose)
- `supine_to_stance` (official public MNN residual recovery, staged for guarded hardware testing)
- `walk` (official Native SDK T800 walking policy)
- `qualifier_front_kick`
- `qualifier_straight_punch`
- `qualifier_hook_punch`
- `qualifier_jab_left`

The initial state is `idle` and does not auto-transition. Walking and standing
motions can only be entered from `pd_stand`; standing motions return to
`pd_stand` automatically.

The two PD poses come from EngineAI's `urkl_exams` branch at commit
`0d759376cba552b480f267042d5d069ad5d96b50`. They are independent preparation
states. EngineAI's accepted `rl_supine_to_stance` policy is reachable only from
`passive` with `START+D-pad up` and returns automatically to zero-command
`walk`. No prone recovery is exposed.

Official bindings reserve `LB+X` and `LB+Y` for the two PD poses. The custom
hook punch and left jab therefore use `LB+B` and `RB+B`, respectively.
The failed spinning kick, the older custom recovery, and prone recovery have no
reachable state or binding in this restricted graph.

Walking reuses the official `rl_walking_example` runner and model and is bound
to `RB+X`, the combination vacated by the rejected spinning kick.

## Keyboard Control

`tools/virtual_gamepad/t800_keyboard_control` is an optional ARM64 terminal
publisher for the same LCM override consumed by the robot-mode input arbiter.
It must receive a live `task_state` before it accepts input and requires an
explicit `--arm` flag. Run it through an interactive SSH session from the
Windows workstation; no LCM or GUI dependency is required on Windows.

The tool enforces the restricted transition graph and intentionally omits the
spinning kick because that policy did not pass its qualification gate. See
`docs/T800_CONTROL_MAPPING.md` in the parent repository for the complete
physical-gamepad, keyboard, and legacy-controller compatibility table.

## Audio Feedback

The launcher configures a non-blocking UDP feedback target at
`192.168.0.162:45800`. Recognized mode combinations generate `KEY` events and
confirmed task changes generate `STATE` events. The companion Orin service in
`tools/audio_feedback` plays one request tone or a two-tone state confirmation.
Audio loss never blocks the control loop.

## Safety Boundary

This package is staged but must not run alongside the official controller.
`run_robot.sh` refuses to start while any `src_executor` process is active.

Before the first hardware run, all of the following must be true:

1. The robot is secured by a load-rated stand or harness with feet clear.
2. The test area is clear and an operator is holding the working emergency stop.
3. The current joint pose matches the selected motion's initial reference.
4. The official service is deliberately stopped and its process is confirmed absent.
5. Only one short motion is enabled for the first low-risk test.

Do not bypass the process checks in `run_robot.sh`.

## Integrity

`DEPLOYMENT_MANIFEST.sha256` contains hashes for the deployed package. Verify it
from the package root before a test with:

```bash
sha256sum -c DEPLOYMENT_MANIFEST.sha256
```

## Runtime Launcher

The Nezha vendor service runs as root. Use `run_custom_robot_root.sh` after the
official service is deliberately stopped. The launcher checks that no other
`src_executor` exists, sources ROS 2 Humble, sets all independent package paths,
and then replaces itself with the custom executor.

The first controlled initialization and two left-jab smoke cycles ran on
2026-09-05. A later provenance audit showed that those cycles used the older
six-motion joint actor instead of the accepted per-motion jab actor. The
candidate was stopped, corrected, and must pass a new staged startup before its
per-motion hardware result is accepted.
