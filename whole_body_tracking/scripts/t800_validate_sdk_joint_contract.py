#!/usr/bin/env python3
"""Validate positional joint semantics across canonical NPZ and EngineAI SDK configs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "source" / "whole_body_tracking"))

from whole_body_tracking.robots.t800_joint_order import (  # noqa: E402
    T800_JOINT_ORDER_VERSION,
    T800_POLICY_JOINT_NAMES,
    T800_SDK_POLICY_JOINT_NAMES,
)


def scalar_text(value: np.ndarray) -> str:
    return str(value.item() if value.ndim == 0 else value.reshape(-1)[0])


def validate_npz(path: Path) -> list[str]:
    errors: list[str] = []
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in ("joint_pos", "joint_vel", "joint_names", "joint_order_version") if key not in data]
        if missing:
            return [f"missing NPZ keys: {missing}"]
        names = [str(name) for name in data["joint_names"].tolist()]
        version = scalar_text(data["joint_order_version"])
        if names != T800_POLICY_JOINT_NAMES:
            errors.append("joint_names is not T800_POLICY_JOINT_NAMES")
        if version != T800_JOINT_ORDER_VERSION:
            errors.append(f"joint_order_version={version!r}, expected {T800_JOINT_ORDER_VERSION!r}")
        for key in ("joint_pos", "joint_vel"):
            if data[key].ndim != 2 or data[key].shape[1] != len(T800_POLICY_JOINT_NAMES):
                errors.append(f"{key} shape={data[key].shape}, expected [frames, 25]")
    return errors


def validate_sdk_config(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    names = payload.get("joint_names", [])
    if names != T800_SDK_POLICY_JOINT_NAMES:
        errors.append("joint_names is not T800_SDK_POLICY_JOINT_NAMES")
    for key in ("default_joint_pos", "joint_stiffness", "joint_damping", "action_scale"):
        values = payload.get(key, [])
        if len(values) != len(T800_SDK_POLICY_JOINT_NAMES):
            errors.append(f"{key} length={len(values)}, expected 25")
    if not payload.get("policy_file"):
        errors.append("policy_file is missing")
    if not payload.get("trajectory_file_npz"):
        errors.append("trajectory_file_npz is missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, action="append", default=[])
    parser.add_argument("--sdk-config", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not args.npz or not args.sdk_config:
        parser.error("at least one --npz and one --sdk-config are required")

    failed = False
    for kind, paths, validator in (
        ("NPZ", args.npz, validate_npz),
        ("SDK", args.sdk_config, validate_sdk_config),
    ):
        for path in paths:
            resolved = path.expanduser().resolve()
            errors = validator(resolved)
            if errors:
                failed = True
                print(f"[FAIL] {kind} {resolved}")
                for error in errors:
                    print(f"  - {error}")
            else:
                print(f"[OK] {kind} {resolved}")

    if not failed:
        print("[OK] NPZ policy indices and EngineAI SDK deploy indices share the same 25 joint semantics")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
