#!/usr/bin/env python3
"""Validate the static model, trajectory, parameter and state-graph contracts."""

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
DISABLED_MOTIONS = {"qualifier_spinning_kick", "qualifier_recovery_supine", "supine_to_stance"}
RECOVERY_PREP = {"pd_stand_x", "pd_stand_y"}
ACTION_POLICIES = {
    "qualifier_front_kick": "front_kick_policy.mnn",
    "qualifier_hook_punch": "hook_punch_policy.mnn",
    "qualifier_jab_left": "jab_left_policy.mnn",
    "qualifier_straight_punch": "straight_punch_policy.mnn",
}
EXPECTED_KEYS = {
    "idle": ("LB", "START"),
    "passive": ("LB", "RB"),
    "pd_stand": ("LB", "A"),
    "pd_stand_x": ("LB", "X"),
    "pd_stand_y": ("LB", "Y"),
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

    task = load_yaml(CONFIG / "task_motion/qualifier_robot.yaml")
    motions = {entry["motion"]: entry for entry in task["tasks"]}
    assert "auto_transition" not in motions["idle"]
    assert DISABLED_MOTIONS.isdisjoint(motions)
    assert set(motions["pd_stand"]["manual_transition"]) == {
        "passive", *RECOVERY_PREP, *STANDING_MOTIONS
    }
    assert RECOVERY_PREP <= set(motions["passive"]["manual_transition"])
    for name in RECOVERY_PREP:
        other = (RECOVERY_PREP - {name}).pop()
        assert motions[name]["runner"] == [
            {"name": "pd_stand_runner", "enabled": True, "param_tag": name}
        ]
        assert set(motions[name]["manual_transition"]) == {"passive", "pd_stand", other}
        assert DISABLED_MOTIONS.isdisjoint(motions[name]["manual_transition"])
    for name in STANDING_MOTIONS:
        assert motions[name]["auto_transition"] == "pd_stand"
        assert set(motions[name]["manual_transition"]) == {"passive", "pd_stand"}
    actual_keys = {name: tuple(motions[name]["key"]) for name in EXPECTED_KEYS}
    assert actual_keys == EXPECTED_KEYS
    assert len(set(actual_keys.values())) == len(actual_keys), "duplicate gamepad binding"


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
    assert (CONFIG / config["policy_file"]).is_file()
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
