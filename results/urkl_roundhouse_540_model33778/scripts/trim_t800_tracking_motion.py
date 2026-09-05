#!/usr/bin/env python3
"""Extract, stabilize, and rebase a window from a T800 tracking NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FRAME_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
VELOCITY_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")
T800_POLICY_JOINT_NAMES = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L", "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J06_HIP_PITCH_R", "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R", "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW", "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    "J20_SHOULDER_PITCH_R", "J21_SHOULDER_ROLL_R", "J22_SHOULDER_YAW_R",
    "J23_ELBOW_PITCH_R", "J24_ELBOW_YAW_R", "J27_HEAD_PITCH", "J28_HEAD_YAW",
]


def yaw_from_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def rotate_world_vectors(values: np.ndarray, angle: float) -> np.ndarray:
    result = np.array(values, copy=True)
    cosine, sine = np.cos(angle), np.sin(angle)
    x = values[..., 0]
    y = values[..., 1]
    result[..., 0] = cosine * x - sine * y
    result[..., 1] = sine * x + cosine * y
    return result


def add_hold(values: np.ndarray, before: int, after: int, zero: bool) -> np.ndarray:
    first = np.zeros_like(values[:1]) if zero else values[:1]
    last = np.zeros_like(values[-1:]) if zero else values[-1:]
    return np.concatenate((np.repeat(first, before, axis=0), values, np.repeat(last, after, axis=0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True, help="Inclusive frame index.")
    parser.add_argument("--end-frame", type=int, required=True, help="Exclusive frame index.")
    parser.add_argument("--hold-before", type=int, default=0)
    parser.add_argument("--hold-after", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=True) as source:
        payload = {key: np.array(source[key]) for key in source.files}
    frame_count = int(payload["joint_pos"].shape[0])
    joint_names = [str(value) for value in payload["joint_names"].reshape(-1)]
    if joint_names != T800_POLICY_JOINT_NAMES:
        raise ValueError("Input does not use the t800_policy_v1 25-joint order")
    if not 0 <= args.start_frame < args.end_frame <= frame_count:
        raise ValueError(f"Invalid frame window {args.start_frame}:{args.end_frame} for {frame_count} frames")
    if args.hold_before < 0 or args.hold_after < 0:
        raise ValueError("Hold frame counts must be non-negative")

    for key in FRAME_KEYS:
        if key not in payload or payload[key].shape[0] != frame_count:
            raise ValueError(f"Missing or invalid frame array: {key}")
        payload[key] = payload[key][args.start_frame : args.end_frame].copy()

    body_names = [str(value) for value in payload["body_names"].reshape(-1)]
    root_index = body_names.index("LINK_BASE")
    origin_xy = payload["body_pos_w"][0, root_index, :2].copy()
    initial_yaw = yaw_from_wxyz(payload["body_quat_w"][0, root_index])
    payload["body_pos_w"][..., :2] -= origin_xy
    payload["body_pos_w"] = rotate_world_vectors(payload["body_pos_w"], -initial_yaw)
    payload["body_lin_vel_w"] = rotate_world_vectors(payload["body_lin_vel_w"], -initial_yaw)
    payload["body_ang_vel_w"] = rotate_world_vectors(payload["body_ang_vel_w"], -initial_yaw)
    yaw_inverse = np.array(
        [np.cos(initial_yaw / 2.0), 0.0, 0.0, -np.sin(initial_yaw / 2.0)],
        dtype=payload["body_quat_w"].dtype,
    )
    payload["body_quat_w"] = quat_multiply_wxyz(yaw_inverse, payload["body_quat_w"])
    payload["body_quat_w"] /= np.linalg.norm(payload["body_quat_w"], axis=-1, keepdims=True)

    for key in FRAME_KEYS:
        payload[key] = add_hold(
            payload[key],
            before=args.hold_before,
            after=args.hold_after,
            zero=key in VELOCITY_KEYS,
        )

    output_frames = int(payload["joint_pos"].shape[0])
    fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
    payload["trim_source"] = np.asarray(str(args.input.resolve()))
    payload["joint_order_version"] = np.asarray("t800_policy_v1")
    payload["trim_window"] = np.asarray([args.start_frame, args.end_frame], dtype=np.int64)
    payload["trim_hold_frames"] = np.asarray([args.hold_before, args.hold_after], dtype=np.int64)
    payload["trim_rebase_xy"] = origin_xy
    payload["trim_rebase_yaw"] = np.asarray(initial_yaw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "source_frames": frame_count,
        "window": {"start_inclusive": args.start_frame, "end_exclusive": args.end_frame},
        "core_frames": args.end_frame - args.start_frame,
        "hold_frames": {"before": args.hold_before, "after": args.hold_after},
        "output_frames": output_frames,
        "duration_seconds": output_frames / fps,
        "source_origin_xy": origin_xy.tolist(),
        "source_initial_yaw_radians": initial_yaw,
        "first_joint_speed": float(np.linalg.norm(payload["joint_vel"][0])),
        "last_joint_speed": float(np.linalg.norm(payload["joint_vel"][-1])),
        "endpoint_joint_distance": float(
            np.linalg.norm(payload["joint_pos"][args.hold_before] - payload["joint_pos"][-args.hold_after - 1])
        ),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
