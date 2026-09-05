"""Canonical joint-order definitions for T800 motion and policy assets."""

T800_JOINT_ORDER_VERSION = "t800_policy_v1"

# Policy/reference order: left leg, right leg, torso, left arm, right arm, head.
T800_POLICY_JOINT_NAMES = [
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

# Legacy tracking NPZ files were written directly from ArticulationData and use
# the URDF topology order below rather than the policy order above.
T800_LEGACY_ISAAC_STORAGE_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J06_HIP_PITCH_R",
    "J12_TORSO_YAW",
    "J01_HIP_ROLL_L",
    "J07_HIP_ROLL_R",
    "J13_SHOULDER_PITCH_L",
    "J20_SHOULDER_PITCH_R",
    "J27_HEAD_PITCH",
    "J02_HIP_YAW_L",
    "J08_HIP_YAW_R",
    "J14_SHOULDER_ROLL_L",
    "J21_SHOULDER_ROLL_R",
    "J28_HEAD_YAW",
    "J03_KNEE_PITCH_L",
    "J09_KNEE_PITCH_R",
    "J15_SHOULDER_YAW_L",
    "J22_SHOULDER_YAW_R",
    "J04_ANKLE_PITCH_L",
    "J10_ANKLE_PITCH_R",
    "J16_ELBOW_PITCH_L",
    "J23_ELBOW_PITCH_R",
    "J05_ANKLE_ROLL_L",
    "J11_ANKLE_ROLL_R",
    "J17_ELBOW_YAW_L",
    "J24_ELBOW_YAW_R",
]

# EngineAI SDK uses different numeric names for the same semantic order.
T800_SDK_POLICY_JOINT_NAMES = [
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


def joint_reorder_indices(source_names: list[str], target_names: list[str]) -> list[int]:
    """Return source column indices arranged in target-name order."""
    if len(source_names) != len(set(source_names)):
        raise ValueError("Source joint names contain duplicates.")
    missing = [name for name in target_names if name not in source_names]
    extra = [name for name in source_names if name not in target_names]
    if missing or extra:
        raise ValueError(f"Joint-name mismatch: missing={missing}, extra={extra}")
    return [source_names.index(name) for name in target_names]


# Backward-compatible name used by the existing conversion/config code.
T800_DFS_JOINT_NAMES = T800_POLICY_JOINT_NAMES
