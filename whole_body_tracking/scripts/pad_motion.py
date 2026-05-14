#!/usr/bin/env python3
"""Pad a motion NPZ by repeating the last frame.

This is useful for very short strike or kick motions where we want the robot to
hold the finishing pose briefly instead of terminating immediately after the
action. Velocity-like arrays are zeroed in the padded region.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pad a tracking motion NPZ by repeating the last frame.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the source motion npz.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the padded motion npz.")
    parser.add_argument(
        "--pad_frames",
        type=int,
        default=150,
        help="Number of extra frames to append using the last pose. Default: 150.",
    )
    return parser.parse_args()


def should_zero_padded_values(key: str) -> bool:
    lowered = key.lower()
    return "vel" in lowered or "ang" in lowered


def pad_motion(input_file: Path, output_file: Path, pad_frames: int) -> None:
    print(f"[INFO] Loading {input_file}")
    data = np.load(str(input_file))
    new_data: dict[str, np.ndarray] = {}

    for key in data.files:
        arr = data[key]
        if isinstance(arr, np.ndarray) and arr.ndim > 0 and arr.shape[0] > 1:
            last_frame = arr[-1:]
            pad_array = np.repeat(last_frame, pad_frames, axis=0)
            if should_zero_padded_values(key):
                pad_array = np.zeros_like(pad_array)
            new_data[key] = np.concatenate([arr, pad_array], axis=0)
            print(f"[INFO] Padded {key}: {arr.shape} -> {new_data[key].shape}")
        else:
            new_data[key] = arr

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_file), **new_data)
    print(f"[OK] Saved padded motion to {output_file}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_file).expanduser().resolve()
    output_path = Path(args.output_file).expanduser().resolve()
    if not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}")
        return 2
    if args.pad_frames < 1:
        print("[ERROR] --pad_frames must be >= 1")
        return 3

    pad_motion(input_path, output_path, args.pad_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
