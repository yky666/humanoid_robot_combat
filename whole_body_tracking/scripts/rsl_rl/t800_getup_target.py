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
    names = payload.get("policy_joint_names")
    if isinstance(names, list) and list(names) != list(T800_POLICY_JOINT_NAMES):
        raise ValueError(
            f"{resolved} policy_joint_names do not match T800_POLICY_JOINT_NAMES"
        )
    return [float(value) for value in values], str(resolved)


def _set_term_target(container, term_name: str, target_joint_pos: list[float]) -> None:
    term = getattr(container, term_name, None)
    if term is not None and hasattr(term, "params") and "target_joint_pos" in term.params:
        term.params["target_joint_pos"] = target_joint_pos


def _set_all_targets_in_container(container, target_joint_pos: list[float]) -> list[str]:
    updated: list[str] = []
    if container is None:
        return updated
    for term_name in dir(container):
        if term_name.startswith("_"):
            continue
        term = getattr(container, term_name, None)
        if term is not None and hasattr(term, "params") and isinstance(getattr(term, "params", None), dict):
            if "target_joint_pos" in term.params:
                term.params["target_joint_pos"] = target_joint_pos
                updated.append(term_name)
    return updated


def apply_getup_target(env_cfg, target_joint_pos: list[float]) -> list[str]:
    if not hasattr(env_cfg, "target_joint_pos"):
        raise ValueError("This task does not expose target_joint_pos; --getup_target_json is only for direct get-up tasks.")
    env_cfg.target_joint_pos = target_joint_pos

    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    if robot_cfg is not None:
        for joint_name, joint_pos in zip(T800_POLICY_JOINT_NAMES, target_joint_pos, strict=True):
            robot_cfg.init_state.joint_pos[joint_name] = joint_pos

    updated: list[str] = []
    observations = getattr(env_cfg, "observations", None)
    if observations is not None:
        for group_name in ("policy", "critic"):
            group = getattr(observations, group_name, None)
            for name in _set_all_targets_in_container(group, target_joint_pos):
                updated.append(f"obs.{group_name}.{name}")

    rewards = getattr(env_cfg, "rewards", None)
    for name in _set_all_targets_in_container(rewards, target_joint_pos):
        updated.append(f"rew.{name}")
    return updated


def apply_getup_target_json(env_cfg, path: str | Path) -> str:
    target_joint_pos, source = load_getup_target_json(path)
    updated = apply_getup_target(env_cfg, target_joint_pos)
    print(f"[INFO] Applied get-up target to {len(updated)} terms: {', '.join(updated) if updated else '(none)'}")
    return source
