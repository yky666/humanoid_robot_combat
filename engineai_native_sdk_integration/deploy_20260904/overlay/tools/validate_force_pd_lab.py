#!/usr/bin/env python3
"""Validate the isolated opt-in force-PD hardware-lab configuration."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "assets/config/t800"
EXPECTED_KEYS = {
    "idle": ("LB", "START"),
    "passive": ("LB", "RB"),
    "pd_stand": ("LB", "A"),
    "pd_stand_x": ("LB", "X"),
    "pd_stand_y": ("LB", "Y"),
}


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main() -> None:
    options = load_yaml(CONFIG / "global_options/force_pd_lab.yaml")
    assert options == {"strict_motion_check": False}

    task = load_yaml(CONFIG / "task_motion/force_pd_lab.yaml")
    assert task["cpu"] == {"number": 3, "priority": 10}
    assert set(task) == {"cpu", "tasks"}
    motions = {entry["motion"]: entry for entry in task["tasks"]}
    assert set(motions) == set(EXPECTED_KEYS)
    assert {name: tuple(motions[name]["key"]) for name in motions} == EXPECTED_KEYS
    assert set(motions["passive"]["manual_transition"]) == {
        "idle", "pd_stand", "pd_stand_x", "pd_stand_y"
    }
    assert set(motions["pd_stand"]["manual_transition"]) == {
        "passive", "pd_stand_x", "pd_stand_y"
    }
    for name in ("pd_stand", "pd_stand_x", "pd_stand_y"):
        assert motions[name]["runner"] == [
            {"name": "pd_stand_runner", "enabled": True, "param_tag": name}
        ]
        assert "auto_transition" not in motions[name]

    print("force-PD lab configuration validation passed")


if __name__ == "__main__":
    main()
