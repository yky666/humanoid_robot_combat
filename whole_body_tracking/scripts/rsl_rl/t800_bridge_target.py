"""Helpers for applying T800 supine-bridge full-state targets."""

from __future__ import annotations

import json
from pathlib import Path

from whole_body_tracking.robots.t800_joint_order import T800_POLICY_JOINT_NAMES


def load_bridge_target_json(path: str | Path) -> tuple[dict, str]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    joints = payload.get("target_joint_pos")
    if not isinstance(joints, list) or len(joints) != len(T800_POLICY_JOINT_NAMES):
        raise ValueError(
            f"{resolved} needs target_joint_pos length {len(T800_POLICY_JOINT_NAMES)}"
        )
    names = payload.get("policy_joint_names")
    if isinstance(names, list) and list(names) != list(T800_POLICY_JOINT_NAMES):
        raise ValueError(f"{resolved} policy_joint_names mismatch")
    for key in ("target_base_pos", "target_base_quat_wxyz"):
        if key not in payload or not isinstance(payload[key], list):
            raise ValueError(f"{resolved} missing {key}")
    if len(payload["target_base_pos"]) != 3:
        raise ValueError("target_base_pos must have length 3")
    if len(payload["target_base_quat_wxyz"]) != 4:
        raise ValueError("target_base_quat_wxyz must have length 4 (wxyz)")
    return payload, str(resolved)


def _set_param(container, key: str, value) -> list[str]:
    updated: list[str] = []
    if container is None:
        return updated
    for term_name in dir(container):
        if term_name.startswith("_"):
            continue
        term = getattr(container, term_name, None)
        params = getattr(term, "params", None)
        if isinstance(params, dict) and key in params:
            params[key] = value
            updated.append(term_name)
    return updated


def apply_bridge_target(env_cfg, payload: dict) -> list[str]:
    required = ("target_joint_pos", "target_base_pos", "target_base_quat_wxyz")
    for key in required:
        if not hasattr(env_cfg, key):
            raise ValueError(
                f"Task does not expose {key}; --bridge_target_json is only for bridge tasks."
            )

    joints = [float(v) for v in payload["target_joint_pos"]]
    base_pos = [float(v) for v in payload["target_base_pos"]]
    base_quat = [float(v) for v in payload["target_base_quat_wxyz"]]
    target_height = float(payload.get("target_height", base_pos[2]))

    env_cfg.target_joint_pos = joints
    env_cfg.target_base_pos = base_pos
    env_cfg.target_base_quat_wxyz = base_quat
    if hasattr(env_cfg, "target_height"):
        env_cfg.target_height = target_height

    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    if robot_cfg is not None:
        for joint_name, joint_pos in zip(T800_POLICY_JOINT_NAMES, joints, strict=True):
            robot_cfg.init_state.joint_pos[joint_name] = joint_pos

    updated: list[str] = []
    observations = getattr(env_cfg, "observations", None)
    rewards = getattr(env_cfg, "rewards", None)
    for group_name in ("policy", "critic"):
        group = getattr(observations, group_name, None) if observations is not None else None
        for key, value in (
            ("target_joint_pos", joints),
            ("target_base_pos", base_pos),
            ("target_base_quat_wxyz", base_quat),
            ("target_height", target_height),
        ):
            for name in _set_param(group, key, value):
                updated.append(f"obs.{group_name}.{name}.{key}")
    for key, value in (
        ("target_joint_pos", joints),
        ("target_base_pos", base_pos),
        ("target_base_quat_wxyz", base_quat),
        ("target_height", target_height),
    ):
        for name in _set_param(rewards, key, value):
            updated.append(f"rew.{name}.{key}")
    return updated


def apply_bridge_target_json(env_cfg, path: str | Path) -> str:
    payload, source = load_bridge_target_json(path)
    updated = apply_bridge_target(env_cfg, payload)
    print(
        f"[INFO] Applied bridge full-state target to {len(updated)} params "
        f"from {source}"
    )
    return source
