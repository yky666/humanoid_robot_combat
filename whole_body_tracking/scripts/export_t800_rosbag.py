#!/usr/bin/env python3
"""Export an EngineAI T800 ROS 2 bag to a uniformly sampled training capture NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_T800_MJCF = Path(__file__).resolve().parents[2] / "GMR" / "assets" / "t800" / "t800.xml"


T800_SDK_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R",
    "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R",
    "J22_ELBOW_YAW_R",
    "J23_HEAD_PITCH",
    "J24_HEAD_YAW",
]

T800_POLICY_JOINT_NAMES = [
    *T800_SDK_JOINT_NAMES[:18],
    "J20_SHOULDER_PITCH_R",
    "J21_SHOULDER_ROLL_R",
    "J22_SHOULDER_YAW_R",
    "J23_ELBOW_PITCH_R",
    "J24_ELBOW_YAW_R",
    "J27_HEAD_PITCH",
    "J28_HEAD_YAW",
]

TOPICS = {
    "joint_state": "/hardware/joint_state",
    "joint_command": "/hardware/joint_command_feedback",
    "motor_state": "/hardware/motor_state",
    "motor_command": "/hardware/motor_command",
    "imu": "/hardware/imu_info",
    "motor_debug": "/hardware/motor_debug",
    "power": "/hardware/power_info",
    "gamepad": "/hardware/gamepad_keys",
    "motion_state": "/motion/motion_state",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="ROS 2 bag directory")
    parser.add_argument("--output", type=Path, required=True, help="Aligned capture NPZ path")
    parser.add_argument("--output-hz", type=float, default=50.0, help="Uniform output rate")
    parser.add_argument("--trim-start", type=float, default=0.0, help="Seconds removed from the beginning")
    parser.add_argument("--trim-end", type=float, default=0.0, help="Seconds removed from the end")
    parser.add_argument(
        "--base-topic",
        default=None,
        help="Optional LinkInfo, Odometry, PoseStamped, or TF topic containing the world-frame base pose",
    )
    parser.add_argument("--base-frame", default="LINK_BASE", help="Child frame selected when --base-topic is /tf")
    parser.add_argument(
        "--root-reconstruction",
        choices=("auto", "sensor", "imu_floor", "none"),
        default="auto",
        help="Root-pose source. Auto uses a base topic, otherwise IMU orientation plus per-frame floor alignment.",
    )
    parser.add_argument("--t800-mjcf", type=Path, default=DEFAULT_T800_MJCF)
    parser.add_argument("--floor-height", type=float, default=0.0)
    parser.add_argument("--floor-clearance", type=float, default=0.005)
    parser.add_argument(
        "--motion-source-output",
        type=Path,
        default=None,
        help="Optionally write a slim root_pos/root_rot/dof_pos copy for t800_csv_to_npz.py",
    )
    return parser.parse_args()


def message_pose(message: Any) -> Any | None:
    pose = getattr(message, "pose", None)
    if pose is None:
        return None
    return getattr(pose, "pose", pose)


def message_twist(message: Any) -> Any | None:
    twist = getattr(message, "twist", None)
    if twist is None:
        return None
    return getattr(twist, "twist", twist)


def message_transform(message: Any, child_frame: str) -> Any | None:
    normalized = child_frame.lstrip("/")
    for transform in getattr(message, "transforms", []):
        if transform.child_frame_id.lstrip("/") == normalized:
            return transform
    return None


def vector3(value: Any) -> np.ndarray:
    return np.asarray([value.x, value.y, value.z], dtype=np.float64)


def quaternion_xyzw(value: Any) -> np.ndarray:
    return np.asarray([value.x, value.y, value.z, value.w], dtype=np.float64)


def geom_bottom_z(model: Any, data: Any, geom_id: int) -> float:
    import mujoco as mj

    geom_type = int(model.geom_type[geom_id])
    position = data.geom_xpos[geom_id]
    size = model.geom_size[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    if geom_type == mj.mjtGeom.mjGEOM_SPHERE:
        return float(position[2] - size[0])
    if geom_type == mj.mjtGeom.mjGEOM_BOX:
        return float(position[2] - np.abs(rotation[2, :]).dot(size[:3]))
    if geom_type in (mj.mjtGeom.mjGEOM_CAPSULE, mj.mjtGeom.mjGEOM_CYLINDER):
        local_extent = np.asarray([size[0], size[0], size[1]], dtype=np.float64)
        return float(position[2] - np.abs(rotation[2, :]).dot(local_extent))
    if geom_type == mj.mjtGeom.mjGEOM_PLANE:
        return float("inf")
    return float(position[2] - float(np.max(size)))


def reconstruct_floor_aligned_root(
    root_rot_xyzw: np.ndarray,
    joint_pos: np.ndarray,
    model_path: Path,
    floor_height: float,
    floor_clearance: float,
) -> np.ndarray:
    try:
        import mujoco as mj
    except ImportError as error:
        raise RuntimeError("imu_floor reconstruction requires the Python mujoco package") from error

    model = mj.MjModel.from_xml_path(str(model_path.expanduser().resolve()))
    if model.nq != joint_pos.shape[1] + 7:
        raise ValueError(f"T800 MJCF expects {model.nq - 7} joints, capture has {joint_pos.shape[1]}")
    data = mj.MjData(model)
    root_pos = np.zeros((joint_pos.shape[0], 3), dtype=np.float32)
    target_bottom_z = floor_height + floor_clearance
    for frame_index in range(joint_pos.shape[0]):
        data.qpos[:3] = 0.0
        data.qpos[3:7] = root_rot_xyzw[frame_index, [3, 0, 1, 2]]
        data.qpos[7:] = joint_pos[frame_index]
        mj.mj_forward(model, data)
        minimum_z = min(geom_bottom_z(model, data, geom_id) for geom_id in range(model.ngeom))
        root_pos[frame_index, 2] = target_bottom_z - minimum_z
    return root_pos


def consistent_samples(
    samples: list[tuple[int, Any]], getter: Callable[[Any], Any], name: str
) -> tuple[np.ndarray, np.ndarray]:
    times: list[int] = []
    values: list[np.ndarray] = []
    width: int | None = None
    for timestamp_ns, message in samples:
        try:
            value = np.asarray(getter(message), dtype=np.float64).reshape(-1)
        except (AttributeError, TypeError, ValueError):
            continue
        if width is None:
            width = value.size
        if value.size != width:
            raise ValueError(f"{name} changed width from {width} to {value.size}")
        times.append(timestamp_ns)
        values.append(value)
    if not values:
        return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float64)
    return np.asarray(times, dtype=np.int64), np.stack(values)


def deduplicate(times_ns: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if times_ns.size < 2:
        return times_ns, values
    keep = np.r_[times_ns[1:] != times_ns[:-1], True]
    return times_ns[keep], values[keep]


def interpolate_matrix(times_ns: np.ndarray, values: np.ndarray, timeline_ns: np.ndarray) -> np.ndarray:
    times_ns, values = deduplicate(times_ns, values)
    if times_ns.size == 0:
        return np.empty((timeline_ns.size, 0), dtype=np.float32)
    if times_ns.size == 1:
        return np.repeat(values.astype(np.float32), timeline_ns.size, axis=0)
    result = np.empty((timeline_ns.size, values.shape[1]), dtype=np.float64)
    relative_source = (times_ns - timeline_ns[0]) * 1e-9
    relative_target = (timeline_ns - timeline_ns[0]) * 1e-9
    for column in range(values.shape[1]):
        result[:, column] = np.interp(relative_target, relative_source, values[:, column])
    return result.astype(np.float32)


def interpolate_quaternion(times_ns: np.ndarray, values: np.ndarray, timeline_ns: np.ndarray) -> np.ndarray:
    values = values.copy()
    for index in range(1, len(values)):
        if np.dot(values[index - 1], values[index]) < 0.0:
            values[index] *= -1.0
    result = interpolate_matrix(times_ns, values, timeline_ns).astype(np.float64)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result /= np.maximum(norms, 1e-12)
    return result.astype(np.float32)


def hold_values(
    samples: list[tuple[int, Any]], getter: Callable[[Any], Any], timeline_ns: np.ndarray, default: Any
) -> np.ndarray:
    if not samples:
        return np.full(timeline_ns.shape, default)
    times_ns = np.asarray([item[0] for item in samples], dtype=np.int64)
    values = np.asarray([getter(item[1]) for item in samples])
    indices = np.searchsorted(times_ns, timeline_ns, side="right") - 1
    indices = np.clip(indices, 0, len(values) - 1)
    return values[indices]


def sample_age_seconds(samples: list[tuple[int, Any]], timeline_ns: np.ndarray) -> np.ndarray:
    if not samples:
        return np.full(timeline_ns.shape, np.inf, dtype=np.float32)
    times_ns = np.asarray([item[0] for item in samples], dtype=np.int64)
    indices = np.searchsorted(times_ns, timeline_ns, side="right") - 1
    valid = indices >= 0
    indices = np.clip(indices, 0, len(times_ns) - 1)
    age = (timeline_ns - times_ns[indices]) * 1e-9
    age[~valid] = np.inf
    return age.astype(np.float32)


def add_matrix(
    output: dict[str, np.ndarray],
    key: str,
    samples: list[tuple[int, Any]],
    getter: Callable[[Any], Any],
    timeline_ns: np.ndarray,
) -> None:
    times_ns, values = consistent_samples(samples, getter, key)
    if values.size:
        output[key] = interpolate_matrix(times_ns, values, timeline_ns)


def read_bag(bag_path: Path) -> tuple[dict[str, list[tuple[int, Any]]], dict[str, str]]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:
        raise RuntimeError("source ROS 2 Humble and the EngineAI interface_protocol workspace first") from error

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    message_classes: dict[str, Any] = {}
    for topic, type_name in topic_types.items():
        try:
            message_classes[topic] = get_message(type_name)
        except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
            continue

    samples: dict[str, list[tuple[int, Any]]] = {topic: [] for topic in topic_types}
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        message_class = message_classes.get(topic)
        if message_class is None:
            continue
        samples[topic].append((timestamp_ns, deserialize_message(serialized, message_class)))
    return samples, topic_types


def choose_base_topic(topic_types: dict[str, str], requested: str | None) -> str | None:
    if requested is not None:
        if requested not in topic_types:
            raise ValueError(f"base topic is not present in bag: {requested}")
        return requested
    supported_types = {
        "interface_protocol/msg/LinkInfo",
        "nav_msgs/msg/Odometry",
        "geometry_msgs/msg/PoseStamped",
    }
    candidates = [
        topic
        for topic, type_name in topic_types.items()
        if type_name in supported_types and any(token in topic.lower() for token in ("base", "root", "odom"))
    ]
    if candidates:
        return sorted(candidates)[0]
    if topic_types.get("/tf") == "tf2_msgs/msg/TFMessage":
        return "/tf"
    return None


def main() -> None:
    args = parse_args()
    if args.output_hz <= 0.0:
        raise ValueError("--output-hz must be positive")
    if args.trim_start < 0.0 or args.trim_end < 0.0:
        raise ValueError("trim values must be non-negative")

    samples, topic_types = read_bag(args.bag.resolve())
    joint_samples = samples.get(TOPICS["joint_state"], [])
    if len(joint_samples) < 2:
        raise RuntimeError(f"bag needs at least two messages on {TOPICS['joint_state']}")

    start_ns = joint_samples[0][0] + round(args.trim_start * 1e9)
    end_ns = joint_samples[-1][0] - round(args.trim_end * 1e9)
    if end_ns <= start_ns:
        raise ValueError("trim range removes the complete recording")
    step_ns = round(1e9 / args.output_hz)
    timeline_ns = np.arange(start_ns, end_ns + 1, step_ns, dtype=np.int64)

    output: dict[str, np.ndarray] = {
        "format_version": np.asarray("t800_real_capture_v1"),
        "source_bag": np.asarray(str(args.bag.resolve())),
        "fps": np.asarray([args.output_hz], dtype=np.float32),
        "time": ((timeline_ns - timeline_ns[0]) * 1e-9).astype(np.float64),
        "time_ns": timeline_ns,
        "joint_names_sdk": np.asarray(T800_SDK_JOINT_NAMES),
        "joint_names_policy": np.asarray(T800_POLICY_JOINT_NAMES),
    }

    add_matrix(output, "joint_pos", joint_samples, lambda message: message.position, timeline_ns)
    add_matrix(output, "joint_vel", joint_samples, lambda message: message.velocity, timeline_ns)
    add_matrix(output, "joint_torque", joint_samples, lambda message: message.torque, timeline_ns)

    joint_command_samples = samples.get(TOPICS["joint_command"], [])
    for key, field in (
        ("joint_cmd_pos", "position"),
        ("joint_cmd_vel", "velocity"),
        ("joint_cmd_tau_ff", "feed_forward_torque"),
        ("joint_cmd_torque", "torque"),
        ("joint_cmd_kp", "stiffness"),
        ("joint_cmd_kd", "damping"),
    ):
        add_matrix(output, key, joint_command_samples, lambda message, name=field: getattr(message, name), timeline_ns)

    motor_state_samples = samples.get(TOPICS["motor_state"], [])
    for key, field in (
        ("motor_pos", "position"),
        ("motor_vel", "velocity"),
        ("motor_torque", "torque"),
    ):
        add_matrix(output, key, motor_state_samples, lambda message, name=field: getattr(message, name), timeline_ns)

    motor_command_samples = samples.get(TOPICS["motor_command"], [])
    for key, field in (
        ("motor_cmd_pos", "position"),
        ("motor_cmd_vel", "velocity"),
        ("motor_cmd_tau_ff", "feed_forward_torque"),
        ("motor_cmd_torque", "torque"),
        ("motor_cmd_kp", "stiffness"),
        ("motor_cmd_kd", "damping"),
    ):
        add_matrix(output, key, motor_command_samples, lambda message, name=field: getattr(message, name), timeline_ns)

    imu_samples = samples.get(TOPICS["imu"], [])
    imu_times, imu_quaternions = consistent_samples(
        imu_samples, lambda message: quaternion_xyzw(message.quaternion), "imu_quat_xyzw"
    )
    if imu_quaternions.size:
        output["imu_quat_xyzw"] = interpolate_quaternion(imu_times, imu_quaternions, timeline_ns)
        output["imu_quat_wxyz"] = output["imu_quat_xyzw"][:, [3, 0, 1, 2]]
    add_matrix(output, "imu_rpy", imu_samples, lambda message: vector3(message.rpy), timeline_ns)
    add_matrix(
        output,
        "imu_linear_acceleration",
        imu_samples,
        lambda message: vector3(message.linear_acceleration),
        timeline_ns,
    )
    add_matrix(
        output,
        "imu_angular_velocity",
        imu_samples,
        lambda message: vector3(message.angular_velocity),
        timeline_ns,
    )

    motor_debug_samples = samples.get(TOPICS["motor_debug"], [])
    for key, field in (
        ("motor_mos_temperature", "mos_temperature"),
        ("motor_temperature", "motor_temperature"),
        ("motor_voltage", "voltage"),
        ("motor_current", "current"),
        ("motor_error_code", "error_code"),
        ("motor_offline", "offline"),
        ("motor_enable", "enable"),
    ):
        add_matrix(output, key, motor_debug_samples, lambda message, name=field: getattr(message, name), timeline_ns)

    power_samples = samples.get(TOPICS["power"], [])
    for key, field in (
        ("power_enable", "enable"),
        ("power_percentage", "percentage"),
        ("power_voltage", "voltage"),
        ("power_current", "current"),
        ("power_current_limit", "current_limit"),
        ("power_error_code", "error_code"),
    ):
        output[key] = hold_values(power_samples, lambda message, name=field: getattr(message, name), timeline_ns, np.nan)

    gamepad_samples = samples.get(TOPICS["gamepad"], [])
    add_matrix(output, "gamepad_digital", gamepad_samples, lambda message: message.digital_states, timeline_ns)
    add_matrix(output, "gamepad_analog", gamepad_samples, lambda message: message.analog_states, timeline_ns)
    output["gamepad_connected"] = hold_values(
        gamepad_samples, lambda message: message.hardware_connected, timeline_ns, False
    )

    motion_samples = samples.get(TOPICS["motion_state"], [])
    output["motion_task"] = hold_values(
        motion_samples, lambda message: message.current_motion_task, timeline_ns, "unknown"
    ).astype(str)
    output["motion_available_transitions_json"] = hold_values(
        motion_samples,
        lambda message: json.dumps(list(message.available_transition_motions), ensure_ascii=False),
        timeline_ns,
        "[]",
    ).astype(str)

    for name, topic in TOPICS.items():
        output[f"age_{name}_s"] = sample_age_seconds(samples.get(topic, []), timeline_ns)

    base_topic = choose_base_topic(topic_types, args.base_topic)
    if base_topic is not None and args.root_reconstruction in ("auto", "sensor"):
        base_samples = samples.get(base_topic, [])
        if topic_types.get(base_topic) == "tf2_msgs/msg/TFMessage":
            pose_times, positions = consistent_samples(
                base_samples,
                lambda message: vector3(message_transform(message, args.base_frame).transform.translation),
                "root_pos",
            )
            quat_times, quaternions = consistent_samples(
                base_samples,
                lambda message: quaternion_xyzw(message_transform(message, args.base_frame).transform.rotation),
                "root_rot",
            )
        else:
            pose_times, positions = consistent_samples(
                base_samples, lambda message: vector3(message_pose(message).position), "root_pos"
            )
            quat_times, quaternions = consistent_samples(
                base_samples, lambda message: quaternion_xyzw(message_pose(message).orientation), "root_rot"
            )
        if positions.size and quaternions.size:
            output["base_topic"] = np.asarray(base_topic)
            output["root_pos"] = interpolate_matrix(pose_times, positions, timeline_ns)
            output["root_rot"] = interpolate_quaternion(quat_times, quaternions, timeline_ns)
        twist_times, linear_velocities = consistent_samples(
            base_samples, lambda message: vector3(message_twist(message).linear), "root_lin_vel"
        )
        angular_times, angular_velocities = consistent_samples(
            base_samples, lambda message: vector3(message_twist(message).angular), "root_ang_vel"
        )
        if linear_velocities.size:
            output["root_lin_vel"] = interpolate_matrix(twist_times, linear_velocities, timeline_ns)
        if angular_velocities.size:
            output["root_ang_vel"] = interpolate_matrix(angular_times, angular_velocities, timeline_ns)

    has_sensor_root = "root_pos" in output and "root_rot" in output and np.isfinite(output["root_pos"]).all()
    if args.root_reconstruction == "sensor" and not has_sensor_root:
        raise RuntimeError("sensor root reconstruction was requested, but no usable dynamic base pose was found")

    use_imu_floor = args.root_reconstruction == "imu_floor" or (
        args.root_reconstruction == "auto" and not has_sensor_root
    )
    if use_imu_floor:
        if "imu_quat_xyzw" not in output:
            raise RuntimeError("IMU quaternion is required when no dynamic base pose is available")
        output["root_rot"] = output["imu_quat_xyzw"].copy()
        output["root_pos"] = reconstruct_floor_aligned_root(
            root_rot_xyzw=output["root_rot"],
            joint_pos=output["joint_pos"],
            model_path=args.t800_mjcf,
            floor_height=args.floor_height,
            floor_clearance=args.floor_clearance,
        )
        output["root_orientation_source"] = np.asarray("imu_quaternion")
        output["root_translation_source"] = np.asarray("zero_xy_per_frame_collision_floor_alignment")
    elif has_sensor_root:
        output["root_orientation_source"] = np.asarray("base_pose_topic")
        output["root_translation_source"] = np.asarray("base_pose_topic")
    else:
        output["root_orientation_source"] = np.asarray("none")
        output["root_translation_source"] = np.asarray("none")

    output["has_training_root_pose"] = np.asarray(
        "root_pos" in output and "root_rot" in output and np.isfinite(output["root_pos"]).all()
    )

    joint_pos = output.get("joint_pos")
    if joint_pos is None or joint_pos.shape[1] != len(T800_SDK_JOINT_NAMES):
        width = 0 if joint_pos is None else joint_pos.shape[1]
        raise ValueError(f"expected {len(T800_SDK_JOINT_NAMES)} T800 joints, got {width}")

    if bool(output["has_training_root_pose"]):
        output["dof_pos"] = output["joint_pos"].copy()
        output["joint_names"] = np.asarray(T800_POLICY_JOINT_NAMES)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(f"[OK] wrote aligned capture: {args.output}")
    print(f"[INFO] frames={timeline_ns.size}, fps={args.output_hz:g}, duration={output['time'][-1]:.3f}s")
    print(f"[INFO] base_pose_topic={base_topic or 'none'}")
    print(f"[INFO] root_translation_source={output['root_translation_source']}")

    if args.motion_source_output is not None:
        if not bool(output["has_training_root_pose"]):
            raise RuntimeError(
                "--motion-source-output requires either sensor or imu_floor root reconstruction"
            )
        args.motion_source_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.motion_source_output,
            fps=np.asarray([args.output_hz], dtype=np.float32),
            root_pos=output["root_pos"],
            root_rot=output["root_rot"],
            dof_pos=output["joint_pos"],
            joint_names=np.asarray(T800_POLICY_JOINT_NAMES),
            source_bag=np.asarray(str(args.bag.resolve())),
        )
        print(f"[OK] wrote motion source: {args.motion_source_output}")


if __name__ == "__main__":
    main()
