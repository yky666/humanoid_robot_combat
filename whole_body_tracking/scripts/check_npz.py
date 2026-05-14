#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


REQUIRED_KEYS = [
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a motion file matches tracking loader requirements.")
    parser.add_argument("motion_file", type=str, help="Path to motion file (.npz expected)")
    args = parser.parse_args()

    motion_path = Path(args.motion_file).expanduser().resolve()
    if not motion_path.is_file():
        print(f"[ERROR] File not found: {motion_path}")
        return 2

    suffix = motion_path.suffix.lower()
    print(f"[INFO] Loading: {motion_path}")
    print(f"[INFO] Suffix: {suffix}")
    if suffix != ".npz":
        print("[WARN] Tracking MotionLoader expects an npz-like container with named arrays.")

    try:
        data = np.load(str(motion_path), allow_pickle=False)
    except Exception as exc:
        print(f"[ERROR] Failed to load with numpy: {exc}")
        return 3

    if not hasattr(data, "files"):
        arr = np.asarray(data)
        print(f"[ERROR] This file is a plain ndarray, not a keyed npz container. shape={arr.shape}, dtype={arr.dtype}")
        return 4

    print("[INFO] Keys and shapes:")
    for key in data.files:
        value = data[key]
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", type(value))
        print(f"  - {key}: shape={shape}, dtype={dtype}")

    missing = [key for key in REQUIRED_KEYS if key not in data.files]
    if missing:
        print(f"[ERROR] Missing required keys: {missing}")
        return 5

    print("[OK] Motion file includes all required keys for tracking MotionLoader.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
