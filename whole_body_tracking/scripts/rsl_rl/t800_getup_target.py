"""Helpers for overriding T800 direct get-up terminal joint targets."""

from __future__ import annotations

import json
from pathlib import Path

from whole_body_tracking.robots.t800_joint_order import T800_POLICY_JOINT_NAMES


def load_getup_target_json(path: str | Path) -> tuple[list[float], str]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    values = payload.get("target_joint_pos")
    if not isinstance(values, list):
        raise ValueError(f"{resolved} does not contain a target_joint_pos list")
    if len(values) != len(T800_POLICY_JOINT_NAMES):
        raise ValueError(
            f"{resolved} target_joint_pos length={len(values)}, expected {len(T800_POLICY_JOINT_NAMES)}"
        )
    return [float(value) for value in values], str(resolved)


def _set_term_target(container, term_name: str, target_joint_pos: list[float]) -> None:
    term = getattr(container, term_name, None)
    if term is not None and hasattr(term, "params") and "target_joint_pos" in term.params:
        term.params["target_joint_pos"] = target_joint_pos


def apply_getup_target(env_cfg, target_joint_pos: list[float]) -> None:
    if not hasattr(env_cfg, "target_joint_pos"):
        raise ValueError("This task does not expose target_joint_pos; --getup_target_json is only for direct get-up tasks.")
    env_cfg.target_joint_pos = target_joint_pos

    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    if robot_cfg is not None:
        for joint_name, joint_pos in zip(T800_POLICY_JOINT_NAMES, target_joint_pos, strict=True):
            robot_cfg.init_state.joint_pos[joint_name] = joint_pos

    observations = getattr(env_cfg, "observations", None)
    if observations is not None:
        for group_name in ("policy", "critic"):
            group = getattr(observations, group_name, None)
            if group is not None:
                _set_term_target(group, "command", target_joint_pos)

    rewards = getattr(env_cfg, "rewards", None)
    if rewards is not None:
        _set_term_target(rewards, "motion_body_pos", target_joint_pos)
        _set_term_target(rewards, "motion_body_ori", target_joint_pos)


def apply_getup_target_json(env_cfg, path: str | Path) -> str:
    target_joint_pos, source = load_getup_target_json(path)
    apply_getup_target(env_cfg, target_joint_pos)
    return source
