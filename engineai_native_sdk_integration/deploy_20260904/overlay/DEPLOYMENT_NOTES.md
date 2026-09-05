# T800 Qualifier Deployment

## Identity

- Package directory: `/home/user/projects/engineai_robotics_qualifier_20260905`
- SDK upstream commit: `335c60e88772c26c7852d0abd6b3c7439037dd8f`
- Target: T800 Nezha controller, ARM64, ROS 2 Humble
- Official package: `/apps/engineai_robotics` (not modified)

## Policy Interface

- Runtime model: `assets/config/t800/rl_qualifier_deploy_20260904/policies/t800_qualifier_joint_policy.mnn`
- Input: `obs`, float32, shape `[1, 140]`
- Output: `actions`, float32, shape `[1, 25]`
- MNN version used for conversion and validation: 2.9.5

The observation order is command, aligned motion anchor position and
orientation, base linear and angular velocity, joint position, joint velocity,
and previous action. The runner rejects invalid dimensions and non-finite data.

## Motions

- `qualifier_front_kick`
- `qualifier_spinning_kick`
- `qualifier_straight_punch`
- `qualifier_hook_punch`
- `qualifier_jab_left`
- `qualifier_recovery_supine`

Standing motions can only be entered from `pd_stand` and return to `pd_stand`.
Recovery can only be entered from `passive` and returns to `passive`.

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

The first controlled initialization succeeded on 2026-09-05. RC02, IMU,
MotorRunner, the 25-joint transform, and ROS 2 initialized without fatal or
error log entries during the observation window. No policy motion was triggered.
