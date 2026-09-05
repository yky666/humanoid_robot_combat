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
    "qualifier_spinning_kick",
    "qualifier_straight_punch",
    "qualifier_hook_punch",
    "qualifier_jab_left",
}
RECOVERY = "qualifier_recovery_supine"


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def validate_state_graph() -> None:
    mode = load_yaml(CONFIG / "mode.yaml")
    assert mode["active_mode"] == "robot"
    robot_scopes = {entry["tag"]: entry["scope"] for entry in mode["mode"]["robot"]}
    assert robot_scopes["motion_task"] == "task_motion/qualifier_robot"

    task = load_yaml(CONFIG / "task_motion/qualifier_robot.yaml")
    motions = {entry["motion"]: entry for entry in task["tasks"]}
    assert set(motions["pd_stand"]["manual_transition"]) == {"passive", *STANDING_MOTIONS}
    assert RECOVERY not in motions["pd_stand"]["manual_transition"]
    assert RECOVERY in motions["passive"]["manual_transition"]
    for name in STANDING_MOTIONS:
        assert motions[name]["auto_transition"] == "pd_stand"
        assert set(motions[name]["manual_transition"]) == {"passive", "pd_stand"}
    assert motions[RECOVERY]["auto_transition"] == "passive"
    assert set(motions[RECOVERY]["manual_transition"]) == {"passive"}


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

    assert config["policy_file"].endswith("t800_qualifier_joint_policy.mnn")
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
    configs = sorted(BUNDLE.glob("qualifier_*.yaml"))
    assert len(configs) == 6
    for config in configs:
        validate_motion(config)
        print(f"validated {config.name}")
    print("qualifier bundle validation passed")


if __name__ == "__main__":
    main()
