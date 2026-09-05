#!/usr/bin/env python3
"""Validate the static model, trajectory, parameter and state-graph contracts."""

import hashlib
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "assets/config/t800"
BUNDLE = CONFIG / "rl_qualifier_deploy_20260904"
OBSERVATIONS = [
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
]
STANDING_MOTIONS = {
    "qualifier_front_kick",
    "qualifier_straight_punch",
    "qualifier_hook_punch",
    "qualifier_jab_left",
}
DISABLED_MOTIONS = {"qualifier_spinning_kick", "qualifier_recovery_supine"}
RECOVERY_PREP = {"pd_stand_x", "pd_stand_y"}
ACTION_POLICIES = {
    "qualifier_front_kick": "front_kick_policy.mnn",
    "qualifier_hook_punch": "hook_punch_policy.mnn",
    "qualifier_jab_left": "jab_left_policy.mnn",
    "qualifier_straight_punch": "straight_punch_policy.mnn",
}
ACTION_POLICY_SHA256 = {
    "front_kick_policy.mnn": "f025f857f074cd5073b6f7abb4eedf677af126c9e934c6706542aa911ca6d8f3",
    "hook_punch_policy.mnn": "f486411792b0744922e195b18c4f2fff09c7ff9ef119a7789af86a27a9195e4b",
    "jab_left_policy.mnn": "a0437216bb8d7d9f339840a804c34ee5c874ab6c193ff1072887aa9a51695697",
    "straight_punch_policy.mnn": "7a863b258a5700942628a7ced386f67d9adfd9eb236dbe14ed082f7b91a4b1fa",
}
EXPECTED_NATIVE_JOINT_NAMES = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L", "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J06_HIP_PITCH_R", "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R", "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW", "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R", "J23_HEAD_PITCH", "J24_HEAD_YAW",
]
EXPECTED_POLICY_PARAMS = {
    "joint_stiffness": [180, 100, 100, 180, 40, 40, 180, 100, 100, 180, 40, 40, 100,
                        40, 40, 40, 40, 50, 40, 40, 40, 40, 50, 50, 50],
    "joint_damping": [5, 3, 3, 5, .3, .3, 5, 3, 3, 5, .3, .3, 3,
                      .3, .3, .3, .3, .3, .3, .3, .3, .3, .3, .3, .3],
    "default_joint_pos": [-.06, 0, 0, .12, -.06, 0, -.06, 0, 0, .12, -.06, 0, 0,
                          0, .15, 0, -.25, 0, 0, -.15, 0, -.25, 0, 0, 0],
    "action_scale": [.5, .2, .2, .5, .5, .2, .5, .2, .2, .5, .5, .2, .2,
                     .2, .2, .05, .2, .05, .2, .2, .05, .2, .05, .2, .2],
}
EXPECTED_KEYS = {
    "idle": ("LB", "START"),
    "passive": ("LB", "RB"),
    "pd_stand": ("LB", "A"),
    "pd_stand_x": ("LB", "X"),
    "pd_stand_y": ("LB", "Y"),
    "supine_to_stance": ("START", "CROSS_X_UP"),
    "walk": ("RB", "X"),
    "qualifier_front_kick": ("RB", "A"),
    "qualifier_straight_punch": ("RB", "Y"),
    "qualifier_hook_punch": ("LB", "B"),
    "qualifier_jab_left": ("RB", "B"),
}


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def validate_state_graph() -> None:
    mode = load_yaml(CONFIG / "mode.yaml")
    assert mode["active_mode"] == "robot"
    robot_scopes = {entry["tag"]: entry["scope"] for entry in mode["mode"]["robot"]}
    assert robot_scopes["motion_task"] == "task_motion/qualifier_robot"
    assert robot_scopes["pd_stand_x"] == "pd_stand/pose_x"
    assert robot_scopes["pd_stand_y"] == "pd_stand/pose_y"
    assert robot_scopes["rl_walking_example"] == "rl_walking_example/default"

    task = load_yaml(CONFIG / "task_motion/qualifier_robot.yaml")
    motions = {entry["motion"]: entry for entry in task["tasks"]}
    assert "auto_transition" not in motions["idle"]
    assert DISABLED_MOTIONS.isdisjoint(motions)
    assert set(motions["pd_stand"]["manual_transition"]) == {
        "passive", "walk", *RECOVERY_PREP, *STANDING_MOTIONS
    }
    assert motions["walk"]["runner"] == [
        {"name": "rl_walking_example_runner", "enabled": True, "param_tag": "rl_walking_example"}
    ]
    assert set(motions["walk"]["manual_transition"]) == {"passive", "pd_stand"}
    assert RECOVERY_PREP <= set(motions["passive"]["manual_transition"])
    assert "supine_to_stance" not in motions["passive"]["manual_transition"]
    for name in RECOVERY_PREP:
        other = (RECOVERY_PREP - {name}).pop()
        assert motions[name]["runner"] == [
            {"name": "pd_stand_runner", "enabled": True, "param_tag": name}
        ]
        assert set(motions[name]["manual_transition"]) == {"passive", "pd_stand", other}
        assert DISABLED_MOTIONS.isdisjoint(motions[name]["manual_transition"])
    assert motions["supine_to_stance"]["runner"] == [
        {"name": "rl_mimic_trajectory_runner", "enabled": True, "param_tag": "rl_supine_to_stance"}
    ]
    assert set(motions["supine_to_stance"]["manual_transition"]) == {"passive"}
    assert motions["supine_to_stance"]["auto_transition"] == "walk"
    for name in STANDING_MOTIONS:
        assert motions[name]["auto_transition"] == "pd_stand"
        assert set(motions[name]["manual_transition"]) == {"passive", "pd_stand"}
    actual_keys = {name: tuple(motions[name]["key"]) for name in EXPECTED_KEYS}
    assert actual_keys == EXPECTED_KEYS
    assert len(set(actual_keys.values())) == len(actual_keys), "duplicate gamepad binding"

    walking_policy = CONFIG / "rl_walking_example/policy/t800_260618_165257_30000.mnn"
    if walking_policy.exists():
        assert hashlib.sha256(walking_policy.read_bytes()).hexdigest() == (
            "cbcb90f86dbb2fde39bdc5a25c8d0530d5c79c7a8f84b1f90863d8c9065b6427"
        )

    recovery_policy = CONFIG / "rl_supine_to_stance/policy/T800_supine_to_stance.mnn"
    recovery_trajectory = CONFIG / "rl_supine_to_stance/trajectory/T800_supine_to_stance.npy"
    assert recovery_policy.is_file() == recovery_trajectory.is_file()
    if recovery_policy.is_file():
        assert hashlib.sha256(recovery_policy.read_bytes()).hexdigest() == (
            "deb9974b1f4f4a7e77801f8c9c6e77f599caab0ca4dd7709fe0bae55870e0e86"
        )
        assert hashlib.sha256(recovery_trajectory.read_bytes()).hexdigest() == (
            "c2f19c164093701311634024eb27999fed4631a00d38d507f8aa306ee138c161"
        )


def validate_pd_pose(name: str) -> None:
    config = load_yaml(CONFIG / f"pd_stand/{name}.yaml")
    for key in ("desired_joint_position", "stiffness", "damping"):
        values = np.asarray([value for group in config[key] for value in group], dtype=np.float64)
        assert values.shape == (25,) and np.isfinite(values).all(), (name, key)
    assert config["duration"] == 3.0
    assert config["initial_joint_position_bias_threshold"] == 4.0


def validate_motion(path: Path) -> None:
    config = load_yaml(path)
    assert config["observation_names"] == OBSERVATIONS
    assert config["observation_history_lengths"] == [1] * len(OBSERVATIONS)
    assert config["expected_observation_dim"] == 140
    assert len(config["joint_names"]) == len(set(config["joint_names"])) == 25
    for key in ("joint_stiffness", "joint_damping", "default_joint_pos", "action_scale"):
        values = np.asarray(config[key], dtype=np.float64)
        assert values.shape == (25,) and np.isfinite(values).all(), (path, key)
    assert 0 < config["action_clip"] <= 1.0
    assert config["transition_time"] >= 3.0
    assert 0 <= config["joint_limit_margin"] <= 0.05

    expected_policy = ACTION_POLICIES.get(path.stem)
    if expected_policy:
        assert Path(config["policy_file"]).name == expected_policy
        assert config["joint_names"] == EXPECTED_NATIVE_JOINT_NAMES
        for key, expected in EXPECTED_POLICY_PARAMS.items():
            assert np.allclose(config[key], expected), (path, key)
    policy_path = CONFIG / config["policy_file"]
    assert policy_path.is_file()
    if expected_policy:
        assert hashlib.sha256(policy_path.read_bytes()).hexdigest() == ACTION_POLICY_SHA256[expected_policy]
    trajectory_path = CONFIG / config["trajectory_file_npz"]
    assert trajectory_path.is_file()
    with np.load(trajectory_path) as trajectory:
        for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w"):
            assert key in trajectory.files
            assert trajectory[key].dtype == np.float32
            assert np.isfinite(trajectory[key]).all()
        frame_count = trajectory["joint_pos"].shape[0]
        assert frame_count >= 2
        assert trajectory["joint_pos"].shape == trajectory["joint_vel"].shape == (frame_count, 25)
        assert trajectory["body_pos_w"].shape[0] == trajectory["body_quat_w"].shape[0] == frame_count
        assert trajectory["body_pos_w"].shape[-1] == 3
        assert trajectory["body_quat_w"].shape[-1] == 4


def main() -> None:
    validate_state_graph()
    validate_pd_pose("pose_x")
    validate_pd_pose("pose_y")
    configs = sorted(BUNDLE.glob("qualifier_*.yaml"))
    assert len(configs) == 6
    for config in configs:
        validate_motion(config)
        print(f"validated {config.name}")
    print("qualifier bundle validation passed")


if __name__ == "__main__":
    main()
