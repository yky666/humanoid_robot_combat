"""Framework-independent T800 guard and deployment contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

T800_TRAINING_JOINT_NAMES = [
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
    "J20_SHOULDER_PITCH_R",
    "J21_SHOULDER_ROLL_R",
    "J22_SHOULDER_YAW_R",
    "J23_ELBOW_PITCH_R",
    "J24_ELBOW_YAW_R",
    "J27_HEAD_PITCH",
    "J28_HEAD_YAW",
]
SDK_ACTIVE_JOINT_NAMES = T800_TRAINING_JOINT_NAMES[:12] + [
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
]
GUARD_JOINT_NAMES = SDK_ACTIVE_JOINT_NAMES[12:]
TRAINING_GUARD_JOINT_NAMES = (
    T800_TRAINING_JOINT_NAMES[13:18] + T800_TRAINING_JOINT_NAMES[18:23]
)

# TienKung expert files use descriptive ``*_joint`` names.  Only this common
# lower-body subset is eligible for transfer; TienKung arm coordinates are
# intentionally discarded before the fixed T800 guard is applied.
TIENKUNG_LOWER_BODY_JOINT_NAMES = [
    "hip_pitch_l_joint",
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_pitch_r_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
]
TIENKUNG_TO_T800_LOWER_BODY = dict(
    zip(TIENKUNG_LOWER_BODY_JOINT_NAMES, T800_TRAINING_JOINT_NAMES[:12])
)

# TienKung crouch kept only as a historical constant. T800 locomotion uses the
# official walk leg defaults plus the measured baoquan upper-body pose.
LOWER_BODY_CROUCH_POSITION = [
    -0.20,
    0.0,
    0.0,
    0.45,
    -0.20,
    0.0,
    -0.20,
    0.0,
    0.0,
    0.45,
    -0.20,
    0.0,
]
CROUCH_ROOT_HEIGHT = 0.77

# Fighting crouch: keep extracted baoquan hip/knee depth and upper body, but
# square the legs so the soles sit on the ground. The raw 1s-tail extract had
# ~0.27 rad hip yaw and a 0.75 m spawn, which put the 2 cm sole boxes ~19 cm
# below z=0 (visible foot penetration) and twisted the feet off the sagittal
# plane. URDF FK of serial_t800.urdf: hip=-0.63, knee=0.79, ankle=-0.175,
# hip_roll=+/-0.10 -> sole zmin=-0.943 at pelvis z=0, so spawn at 0.95 m.
_BAOQUAN_HIP_PITCH = -0.63
_BAOQUAN_KNEE_PITCH = 0.79
_BAOQUAN_HIP_ROLL = 0.10
_BAOQUAN_ANKLE_PITCH = -0.175
T800_BAOQUAN_POLICY_POSE = {
    "J00_HIP_PITCH_L": _BAOQUAN_HIP_PITCH,
    "J01_HIP_ROLL_L": _BAOQUAN_HIP_ROLL,
    "J02_HIP_YAW_L": 0.0,
    "J03_KNEE_PITCH_L": _BAOQUAN_KNEE_PITCH,
    "J04_ANKLE_PITCH_L": _BAOQUAN_ANKLE_PITCH,
    "J05_ANKLE_ROLL_L": -_BAOQUAN_HIP_ROLL,
    "J06_HIP_PITCH_R": _BAOQUAN_HIP_PITCH,
    "J07_HIP_ROLL_R": -_BAOQUAN_HIP_ROLL,
    "J08_HIP_YAW_R": 0.0,
    "J09_KNEE_PITCH_R": _BAOQUAN_KNEE_PITCH,
    "J10_ANKLE_PITCH_R": _BAOQUAN_ANKLE_PITCH,
    "J11_ANKLE_ROLL_R": _BAOQUAN_HIP_ROLL,
    "J12_TORSO_YAW": -0.065857,
    "J13_SHOULDER_PITCH_L": -0.688721,
    "J14_SHOULDER_ROLL_L": 0.902954,
    "J15_SHOULDER_YAW_L": -0.142029,
    "J16_ELBOW_PITCH_L": -1.917786,
    "J17_ELBOW_YAW_L": -0.218201,
    "J20_SHOULDER_PITCH_R": -1.19574,
    "J21_SHOULDER_ROLL_R": -0.167419,
    "J22_SHOULDER_YAW_R": 0.582397,
    "J23_ELBOW_PITCH_R": -1.878113,
    "J24_ELBOW_YAW_R": 0.199158,
    "J27_HEAD_PITCH": 0.245972,
    "J28_HEAD_YAW": 0.588745,
}
T800_BAOQUAN_LEG_POSITION = [
    T800_BAOQUAN_POLICY_POSE[name] for name in T800_TRAINING_JOINT_NAMES[:12]
]
T800_STAND_LEG_POSITION = T800_BAOQUAN_LEG_POSITION
T800_BAOQUAN_ROOT_HEIGHT = 0.95
T800_STAND_ROOT_HEIGHT = T800_BAOQUAN_ROOT_HEIGHT
T800_BAOQUAN_TORSO_YAW = T800_BAOQUAN_POLICY_POSE["J12_TORSO_YAW"]
T800_BAOQUAN_HEAD_POSITION = [
    T800_BAOQUAN_POLICY_POSE["J27_HEAD_PITCH"],
    T800_BAOQUAN_POLICY_POSE["J28_HEAD_YAW"],
]

_WORKSPACE_GUARD_PATH = (
    Path(__file__).resolve().parents[6]
    / "GMR/general_motion_retargeting/ik_configs/t800_walk_guard_pose.json"
)


def load_guard_pose(
    path: str | Path = _WORKSPACE_GUARD_PATH,
) -> tuple[list[str], list[float]]:
    """Load the auditable guard JSON and validate its basic shape."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    names = payload.get("joint_names")
    position = payload.get("position_rad")
    if (
        names != GUARD_JOINT_NAMES
        or not isinstance(position, list)
        or len(position) != len(GUARD_JOINT_NAMES)
    ):
        raise ValueError(
            "guard pose JSON does not match the canonical T800 ten-joint contract"
        )
    values = [float(value) for value in position]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("guard pose contains non-finite values")
    return list(names), values


def build_training_to_sdk_mapping(
    training_names: Iterable[str] = T800_TRAINING_JOINT_NAMES,
    sdk_names: Iterable[str] = SDK_ACTIVE_JOINT_NAMES,
) -> list[int]:
    """Return SDK output indices in training order, keyed by semantic names."""
    training = list(training_names)
    sdk = list(sdk_names)
    if (
        len(training) != 25
        or len(sdk) != 22
        or len(set(training)) != len(training)
        or len(set(sdk)) != len(sdk)
    ):
        raise ValueError(
            "T800 training/SDK joint contracts must contain unique 25/22 names"
        )
    # The right-arm execution names are semantically shifted by two numbers.
    aliases = {
        "J18_SHOULDER_PITCH_R": "J20_SHOULDER_PITCH_R",
        "J19_SHOULDER_ROLL_R": "J21_SHOULDER_ROLL_R",
        "J20_SHOULDER_YAW_R": "J22_SHOULDER_YAW_R",
        "J21_ELBOW_PITCH_R": "J23_ELBOW_PITCH_R",
        "J22_ELBOW_YAW_R": "J24_ELBOW_YAW_R",
    }
    resolved = [aliases.get(name, name) for name in sdk]
    missing = [name for name in resolved if name not in training]
    if missing or len(set(resolved)) != len(resolved):
        raise ValueError(
            f"SDK joints cannot be resolved against training contract: {missing}"
        )
    return [training.index(name) for name in resolved]


def adapt_training_action_to_sdk(action: Iterable[float]) -> list[float]:
    values = list(action)
    if len(values) != 25:
        raise ValueError(
            f"T800 training action must contain 25 values, got {len(values)}"
        )
    mapping = build_training_to_sdk_mapping()
    return [float(values[index]) for index in mapping]


def build_sdk_deployment_metadata() -> dict[str, object]:
    """Return the deployment contract consumed by SDK policy adapters."""
    return {
        "robot": "t800",
        "task": "Tracking-Flat-T800-Fixed-Guard-v0",
        "training_joint_names": list(T800_TRAINING_JOINT_NAMES),
        "sdk_active_joint_names": list(SDK_ACTIVE_JOINT_NAMES),
        "training_to_sdk_indices": build_training_to_sdk_mapping(),
        "guard_joint_names": list(GUARD_JOINT_NAMES),
        "guard_source": "GMR/general_motion_retargeting/ik_configs/t800_walk_guard_pose.json",
        "action_adapter": "25_to_22_semantic",
        "policy_input_dim": 97,
        "policy_output_dim": 25,
        "policy_observation_normalizer": "embedded",
        "sdk_input_contract": (
            "training_observation_dim=97;history_steps=1;no_force_fields"
        ),
        "sdk_observation_sources": {
            "base_lin_vel": (
                "DataStore.base_state_in_world.frame.twist.linear rotated to base"
            ),
            "base_ang_vel": "DataStore.imu_info.angular_velocity rotated to base",
            "projected_gravity": "IMU quaternion",
            "velocity_commands": "filtered virtual-gamepad command",
            "joint_pos": "22 SDK joints plus explicit zero torso/head fields",
            "joint_vel": "22 SDK joints plus explicit zero torso/head fields",
            "previous_actions": "25-D training action history",
            "gait_phase": "runner control time, 0.85 s cycle",
            "feet_positions_b": "DataStore.end_state_in_base LINK_FOOT_L/R",
        },
    }
