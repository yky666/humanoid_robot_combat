# T800 Real-Motion Recording

This workflow records official-controller motions without publishing any robot
command. Keep the untouched ROS 2 bag as the source of truth and derive aligned
NPZ files from it afterward.

## Captured Data

The recorder requires these topics:

- `/hardware/joint_state`: measured joint position, velocity, and torque;
- `/hardware/joint_command_feedback`: commanded position, velocity,
  feed-forward torque, resulting torque, stiffness, and damping;
- `/hardware/imu_info`: orientation, RPY, acceleration, and angular velocity.

When available, it also records motor state and command, motor diagnostics,
power, gamepad, motion state, TF, odometry, base/link pose, foot contact, force,
and wrench topics. The capture manifest stores the action label, initial
orientation, host, time, ROS domain, topic names, and message types.

The default output root matches the existing workstation layout:

```text
/home/ubuntu/source/engineai_workspace/src/interface_example/scripts/logs
```

## Record One Trial

Run the recorder in a terminal that has access to the same ROS domain as the
official controller:

```bash
cd /path/to/humanoid_robot_combat/whole_body_tracking

scripts/record_t800_motion.sh \
  --action straight_punch \
  --orientation upright \
  --notes "official controller, trial 01"
```

Wait until rosbag prints `Recording...`, hold the initial pose for at least two
seconds, trigger exactly one official action, and keep recording for at least
two seconds after the final pose settles. Press `Ctrl-C` once. Do not combine
multiple action types in the same bag.

Recommended labels are:

```text
boxing_ready_hold
straight_punch
left_front_kick
supine_to_stance
prone_to_stance
```

Examples for both recovery orientations:

```bash
scripts/record_t800_motion.sh \
  --action supine_to_stance \
  --orientation supine \
  --notes "official recovery, trial 01"

scripts/record_t800_motion.sh \
  --action prone_to_stance \
  --orientation prone \
  --notes "official recovery, trial 01"
```

Use `--duration 20` for an automatic 20-second capture. Repeat each motion at
least 5 to 10 times, preserving each trial separately. Include successful
nominal executions first; collect deliberately perturbed trials only as a
separate dataset with explicit notes.

Camera topics are not selected automatically because their bandwidth can
disturb high-rate control telemetry. Add them explicitly only when needed:

```bash
scripts/record_t800_motion.sh \
  --action straight_punch \
  --orientation upright \
  --extra-topic /camera/color/image_raw \
  --extra-topic /camera/depth/image_raw
```

## Root Reconstruction

The recorder does not require a dynamic base-pose topic. It always preserves
joint, command, torque, and IMU data for upright, supine, and prone motions. It
also automatically includes topics whose names contain `base`, `odom`,
`link_info`, `contact`, `foot`, `wrench`, or `force`, plus `/tf` and
`/tf_static` when present.

Before recording, inspect the available topics:

```bash
ros2 topic list -t | sort
```

When a dynamic world-frame base pose exists, the exporter uses it. Otherwise,
the exporter uses the measured IMU quaternion as `root_rot`, normalizes
`root_x/root_y` to zero, and reconstructs `root_z` independently for every frame
so the lowest T800 collision geometry touches the configured floor. This
`imu_floor` reconstruction produces the same `root_pos/root_rot/dof_pos` input
contract accepted by the existing NPZ conversion pipeline. The generated
metadata explicitly distinguishes reconstructed translation from measured
translation.

## Export an Aligned Capture

Source ROS 2 Humble and the EngineAI message workspace, then run:

```bash
python scripts/export_t800_rosbag.py \
  /path/to/logs/20260906_120000_straight_punch_upright \
  --output /path/to/logs/straight_punch_trial01_capture.npz \
  --output-hz 50 \
  --trim-start 1.0 \
  --trim-end 1.0
```

On the training workstation, source ROS 2 and the EngineAI messages and make
sure Python `mujoco` is available. Export the bag and pass the resulting file
to the existing IsaacLab converter:

```bash
python scripts/export_t800_rosbag.py \
  /path/to/logs/20260906_120000_prone_to_stance_prone \
  --output /path/to/logs/prone_trial01_capture.npz \
  --output-hz 50

python scripts/t800_csv_to_npz.py \
  --input_file /path/to/logs/prone_trial01_capture.npz \
  --input_format gmr_npz \
  --output_name prone_trial01_tracking \
  --output_path /path/to/logs/prone_trial01_tracking.npz \
  --output_fps 50 \
  --skip_wandb_upload \
  --headless
```

No base-pose argument is needed in the normal case because
`--root-reconstruction auto` is the default. To force TF input, pass
`--root-reconstruction sensor --base-topic /tf --base-frame LINK_BASE`. To
force joint-plus-IMU reconstruction, pass `--root-reconstruction imu_floor`.

The aligned capture itself contains the GMR-compatible
`fps/root_pos/root_rot/dof_pos` keys, so it can be passed directly to
`t800_csv_to_npz.py`. Use `--motion-source-output` only when a smaller copy
without diagnostic telemetry is useful.

## DB3 and Playback

The `.db3` file is the normal SQLite payload of a ROS 2 bag. It is the raw
telemetry source, not an IsaacLab motion file. A complete bag directory also
contains `metadata.yaml`; use the directory rather than the `.db3` filename:

```bash
ros2 bag info /path/to/recording_directory
ros2 bag play /path/to/recording_directory
```

`ros2 bag play` only republishes the recorded topics. It does not visualize the
robot and it must not be treated as a safe command replay mechanism. To view the
motion in IsaacLab, first export and convert the bag to the tracking NPZ shown
above, then use `scripts/rsl_rl/play_t800.py` with the trained policy or the
repository's motion visualization workflow.

If a recording directory contains only `.db3`, the recorder was usually killed,
the terminal or SSH session was closed, or power was lost before rosbag wrote
its index. Repair it after sourcing the same ROS 2 environment:

```bash
ros2 bag reindex /path/to/recording_directory
ros2 bag info /path/to/recording_directory
```

The recorder now checks for this condition after a normal stop and automatically
runs `ros2 bag reindex` when needed. Always stop recording with `Ctrl-C` once;
do not use `kill -9`, close the SSH terminal, or power off the computer while
rosbag is finalizing.

## Two Replay Paths

There are two different meanings of replay:

1. ROS bag replay for inspection:

   ```bash
   source /opt/ros/humble/setup.bash
   source /home/ubuntu/source/engineai_workspace/install/setup.bash
   ros2 bag info /path/to/logs/20260906_120000_straight_punch_upright
   ros2 bag play /path/to/logs/20260906_120000_straight_punch_upright
   ```

   This republishes recorded telemetry topics with their original timestamps.
   It is useful for checking parsers, plotting, synchronization, and offline
   subscribers. It does not automatically animate IsaacLab, and it should not be
   used as a real-robot actuator command unless a reviewed command subscriber,
   emergency stop, low stiffness, joint-limit guards, and operator procedure are
   in place.

2. Real robot motion execution:

   The deployable path is `record -> export -> convert -> train in simulation ->
   export policy -> deploy through the EngineAI motion runner`. The current
   deployment runner consumes a trained MNN policy plus a reference tracking NPZ:

   - `policy_file`: trained `.mnn` policy;
   - `trajectory_file_npz`: converted tracking `.npz`;
   - `joint_names`, `joint_stiffness`, `joint_damping`, `default_joint_pos`,
     `action_scale`: same order and dimensions used in training;
   - `transition_time`, `max_initial_pose_error`, `joint_limit_margin`,
     `action_clip`, `expected_observation_dim`: deployment guards.

   See `engineai_native_sdk_integration/deploy_20260904/overlay/assets/config/t800/rl_qualifier_deploy_20260904/*.yaml`
   and `engineai_native_sdk_integration/deploy_20260904/overlay/src/runner/rl_dance_example/src/rl_dance_example_runner.cc`.

   A direct open-loop joint trajectory runner can be added later if required,
   but it should read an exported trajectory NPZ and command position/velocity
   under strict guards rather than replaying the whole bag blindly. For recovery
   motions, prefer the trained closed-loop policy path first.

## Training and Deployment

Inspect each generated tracking NPZ visually, validate its joint order and
schema, and train one policy per motion. Boxing motions should include a stable
`boxing_ready -> action -> boxing_ready` transition. Supine and prone recovery
must be trained independently unless prone recovery is intentionally split into
`prone_to_supine` followed by the qualified supine policy.

Measured and commanded torque should be retained for diagnostics, actuator
model calibration, limits, and reward design. Do not replay a measured torque
sequence open loop on the real robot. Deploy only a simulation-qualified,
closed-loop policy with initial-pose, IMU, contact, finite-output, joint-limit,
torque, timeout, and stable-terminal-state guards.

## Training Data Contract

Keep the original bag directory intact. The bag is the audit artifact; all NPZ
files are derived products.

Required bag topics for training conversion:

- `/hardware/joint_state`: measured joint position, velocity, and torque. The
  exporter requires at least two messages and expects 25 T800 joints.
- `/hardware/imu_info`: quaternion, RPY, linear acceleration, and angular
  velocity. This is required when no base pose topic exists.
- `/hardware/joint_command_feedback`: commanded position, velocity, feed-forward
  torque, output torque, stiffness, and damping. This is not required by the
  tracking NPZ loader, but it is important for diagnostics, actuator fitting,
  and checking what the official controller actually asked the robot to do.

Useful optional topics:

- `/hardware/motor_state`, `/hardware/motor_command`, `/hardware/motor_debug`;
- `/hardware/power_info`, `/hardware/gamepad_keys`, `/motion/motion_state`;
- `/tf`, `/tf_static`, odometry, base/root/link pose, foot contact, force, and
  wrench topics when available.

The aligned capture produced by `scripts/export_t800_rosbag.py` contains:

- raw aligned telemetry: `joint_pos`, `joint_vel`, `joint_torque`,
  `joint_cmd_pos`, `joint_cmd_vel`, `joint_cmd_tau_ff`, `joint_cmd_torque`,
  `joint_cmd_kp`, `joint_cmd_kd`, IMU, power, gamepad, and diagnostic age
  arrays;
- training source keys: `fps`, `root_pos`, `root_rot` in xyzw order, and
  `dof_pos`;
- provenance keys: `source_bag`, `format_version`, `joint_names_sdk`,
  `joint_names_policy`, `root_orientation_source`, and
  `root_translation_source`.

The final tracking NPZ produced by `scripts/t800_csv_to_npz.py` is what the
IsaacLab tracking loader consumes. It must contain:

- `fps`;
- `joint_names`;
- `joint_pos` and `joint_vel`, shaped `[T, 25]`;
- `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`.

The tracking loader validates these keys and reorders columns by `joint_names`.
The canonical T800 policy order is defined in
`whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/t800_joint_order.py`.

End-to-end conversion example:

```bash
cd /path/to/humanoid_robot_combat/whole_body_tracking

python scripts/export_t800_rosbag.py \
  /home/ubuntu/source/engineai_workspace/src/interface_example/scripts/logs/20260906_120000_prone_to_stance_prone \
  --output artifacts/real_captures/prone_trial01_capture.npz \
  --output-hz 50 \
  --trim-start 1.0 \
  --trim-end 1.0

python scripts/t800_csv_to_npz.py \
  --input_file artifacts/real_captures/prone_trial01_capture.npz \
  --input_format gmr_npz \
  --output_name prone_trial01_tracking \
  --output_path artifacts/real_captures/prone_trial01_tracking.npz \
  --output_fps 50 \
  --skip_wandb_upload \
  --headless
```

For supine and prone get-up, use `--orientation supine` or `--orientation prone`
while recording. No dynamic base pose is required. If the bag has no base pose,
the exporter uses IMU orientation and per-frame floor alignment from the T800
MJCF to produce `root_pos/root_rot/dof_pos` compatible with the existing
`gmr_npz` conversion path.
