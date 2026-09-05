#!/usr/bin/env python3
"""Validate T800 tracking NPZ joint metadata and optional raw-GMR equivalence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "source" / "whole_body_tracking"))

from whole_body_tracking.robots.t800_joint_order import (  # noqa: E402
    T800_JOINT_ORDER_VERSION,
    T800_POLICY_JOINT_NAMES,
)


def decode_names(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values.tolist()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracking_npz", type=Path)
    parser.add_argument("--raw-gmr", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    args = parser.parse_args()

    path = args.tracking_npz.expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        missing = [
            key
            for key in (
                "fps",
                "joint_names",
                "joint_order_version",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
            )
            if key not in data
        ]
        if missing:
            raise SystemExit(f"{path}: missing keys {missing}")
        names = decode_names(np.asarray(data["joint_names"]).reshape(-1))
        version = str(np.asarray(data["joint_order_version"]).reshape(-1)[0])
        joint_pos = np.asarray(data["joint_pos"])
        joint_vel = np.asarray(data["joint_vel"])

    if names != T800_POLICY_JOINT_NAMES:
        raise SystemExit(f"{path}: non-canonical joint names: {names}")
    if version != T800_JOINT_ORDER_VERSION:
        raise SystemExit(f"{path}: joint_order_version={version!r}")
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(names) or joint_vel.shape != joint_pos.shape:
        raise SystemExit(f"{path}: invalid joint shapes pos={joint_pos.shape} vel={joint_vel.shape}")
    if not np.isfinite(joint_pos).all() or not np.isfinite(joint_vel).all():
        raise SystemExit(f"{path}: non-finite joint values")

    raw_error = None
    if args.raw_gmr is not None:
        with np.load(args.raw_gmr.expanduser().resolve(), allow_pickle=False) as raw:
            raw_pos = np.asarray(raw["dof_pos"])
        raw_error = float(np.max(np.abs(joint_pos[0] - raw_pos[0])))
        if raw_error > args.atol:
            raise SystemExit(f"{path}: first-frame raw-GMR error {raw_error:g} exceeds {args.atol:g}")

    message = f"[OK] {path} order={version} frames={joint_pos.shape[0]} joints={joint_pos.shape[1]}"
    if raw_error is not None:
        message += f" raw_first_frame_max_error={raw_error:g}"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
