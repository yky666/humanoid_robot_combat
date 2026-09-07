#!/usr/bin/env python3
"""Validate a T800 recovery reference from an official PD pose to boxing ready."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


WBT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = WBT_ROOT.parent
sys.path.insert(0, str(WBT_ROOT / "source" / "whole_body_tracking"))

from whole_body_tracking.robots.t800_joint_order import (  # noqa: E402
    T800_JOINT_ORDER_VERSION,
    T800_POLICY_JOINT_NAMES,
)


DEFAULT_PD_ROOT = (
    REPOSITORY_ROOT
    / "engineai_native_sdk_integration"
    / "deploy_20260904"
    / "overlay"
    / "assets"
    / "config"
    / "t800"
    / "pd_stand"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, help="Candidate recovery tracking NPZ")
    parser.add_argument(
        "--orientation",
        choices=("prone", "supine"),
        required=True,
        help="Expected floor orientation and corresponding official preparation pose",
    )
    parser.add_argument(
        "--ready-reference",
        type=Path,
        required=True,
        help="Approved boxing-ready hold tracking NPZ used as the terminal target",
    )
    parser.add_argument("--pose-file", type=Path, default=None, help="Override the official pose_x/pose_y YAML")
    parser.add_argument("--start-hold-frames", type=int, default=10)
    parser.add_argument("--terminal-hold-frames", type=int, default=25)
    parser.add_argument("--start-max-error", type=float, default=0.08, help="Maximum PD start error in radians")
    parser.add_argument("--terminal-max-error", type=float, default=0.08, help="Maximum ready-pose error in radians")
    parser.add_argument("--hold-max-drift", type=float, default=0.05, help="Maximum hold-segment drift in radians")
    parser.add_argument(
        "--max-planar-displacement",
        type=float,
        default=3.5,
        help="Maximum reference root displacement from its initial XY position in metres",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON validation report")
    return parser.parse_args()


def scalar_text(value: np.ndarray) -> str:
    item = value.item() if value.ndim == 0 else value.reshape(-1)[0]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def decode_names(value: np.ndarray) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in value.reshape(-1).tolist()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tracking(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "fps",
            "joint_names",
            "joint_order_version",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "body_names",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path}: missing tracking keys {missing}")
        names = decode_names(np.asarray(data["joint_names"]))
        version = scalar_text(np.asarray(data["joint_order_version"]))
        if names != T800_POLICY_JOINT_NAMES:
            raise ValueError(f"{path}: joint_names is not canonical T800 policy order")
        if version != T800_JOINT_ORDER_VERSION:
            raise ValueError(f"{path}: joint_order_version={version!r}, expected {T800_JOINT_ORDER_VERSION!r}")
        body_names = decode_names(np.asarray(data["body_names"]))
        if not body_names or body_names[0] != "LINK_BASE":
            raise ValueError(f"{path}: body_names must begin with LINK_BASE, got {body_names[:1]}")

        metadata_keys = {"fps", "joint_names", "joint_order_version", "body_names"}
        arrays = {
            key: np.asarray(data[key])
            for key in required
            if key not in metadata_keys
        }
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    joint_pos = arrays["joint_pos"]
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(T800_POLICY_JOINT_NAMES):
        raise ValueError(f"{path}: joint_pos shape must be [frames, 25], got {joint_pos.shape}")
    if arrays["joint_vel"].shape != joint_pos.shape:
        raise ValueError(f"{path}: joint_vel shape {arrays['joint_vel'].shape} does not match joint_pos")
    if arrays["body_pos_w"].ndim != 3 or arrays["body_pos_w"].shape[2] != 3:
        raise ValueError(f"{path}: body_pos_w shape must be [frames, bodies, 3]")
    if arrays["body_quat_w"].ndim != 3 or arrays["body_quat_w"].shape[2] != 4:
        raise ValueError(f"{path}: body_quat_w shape must be [frames, bodies, 4]")
    if arrays["body_quat_w"].shape[:2] != arrays["body_pos_w"].shape[:2]:
        raise ValueError(f"{path}: body position/quaternion dimensions do not match")
    frame_count = joint_pos.shape[0]
    if frame_count < 2 or fps <= 0.0:
        raise ValueError(f"{path}: invalid frames/fps: frames={frame_count}, fps={fps}")
    for key, value in arrays.items():
        if value.shape[0] != frame_count:
            raise ValueError(f"{path}: {key} has {value.shape[0]} frames, expected {frame_count}")
        if not np.isfinite(value).all():
            raise ValueError(f"{path}: {key} contains non-finite values")
    quaternion_norms = np.linalg.norm(arrays["body_quat_w"], axis=-1)
    if float(np.max(np.abs(quaternion_norms - 1.0))) > 1.0e-3:
        raise ValueError(f"{path}: body_quat_w contains non-unit quaternions")
    return {**arrays, "fps": fps}


def load_pd_pose(path: Path) -> np.ndarray:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    groups = payload.get("desired_joint_position")
    if not isinstance(groups, list):
        raise ValueError(f"{path}: desired_joint_position is missing")
    values = np.asarray([item for group in groups for item in group], dtype=np.float64)
    if values.shape != (len(T800_POLICY_JOINT_NAMES),):
        raise ValueError(f"{path}: expected 25 desired joints, got {values.shape}")
    return values


def tail_median(joint_pos: np.ndarray, frames: int) -> tuple[np.ndarray, int]:
    count = min(max(frames, 1), joint_pos.shape[0])
    return np.median(joint_pos[-count:], axis=0), count


def max_hold_drift(segment: np.ndarray) -> float:
    center = np.median(segment, axis=0)
    return float(np.max(np.abs(segment - center)))


def root_planar_displacement(body_pos_w: np.ndarray) -> float:
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w shape must be [frames, bodies, 3], got {body_pos_w.shape}")
    root_xy = body_pos_w[:, 0, :2]
    return float(np.max(np.linalg.norm(root_xy - root_xy[0], axis=1)))


def main() -> int:
    args = parse_args()
    motion_path = args.motion.expanduser().resolve()
    ready_path = args.ready_reference.expanduser().resolve()
    pose_path = (
        args.pose_file.expanduser().resolve()
        if args.pose_file is not None
        else DEFAULT_PD_ROOT / ("pose_x.yaml" if args.orientation == "prone" else "pose_y.yaml")
    )

    errors: list[str] = []
    try:
        motion = load_tracking(motion_path)
        ready = load_tracking(ready_path)
        pd_pose = load_pd_pose(pose_path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        report = {"status": "failed", "errors": [str(error)]}
        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2

    joint_pos = np.asarray(motion["joint_pos"])
    ready_pos = np.asarray(ready["joint_pos"])
    start_count = min(max(args.start_hold_frames, 1), joint_pos.shape[0])
    terminal_target, ready_count = tail_median(ready_pos, args.terminal_hold_frames)
    terminal_pose, terminal_count = tail_median(joint_pos, args.terminal_hold_frames)

    start_error = np.abs(joint_pos[0] - pd_pose)
    terminal_error = np.abs(terminal_pose - terminal_target)
    start_drift = max_hold_drift(joint_pos[:start_count])
    terminal_drift = max_hold_drift(joint_pos[-terminal_count:])
    ready_drift = max_hold_drift(ready_pos[-ready_count:])
    displacement = root_planar_displacement(np.asarray(motion["body_pos_w"]))

    gates = {
        "start_pose": float(np.max(start_error)) <= args.start_max_error,
        "start_hold": start_drift <= args.hold_max_drift,
        "terminal_pose": float(np.max(terminal_error)) <= args.terminal_max_error,
        "terminal_hold": terminal_drift <= args.hold_max_drift,
        "ready_reference_hold": ready_drift <= args.hold_max_drift,
        "planar_displacement": displacement <= args.max_planar_displacement,
    }
    for name, passed in gates.items():
        if not passed:
            errors.append(f"gate failed: {name}")

    worst_start_index = int(np.argmax(start_error))
    worst_terminal_index = int(np.argmax(terminal_error))
    report = {
        "status": "passed" if not errors else "failed",
        "orientation": args.orientation,
        "motion": str(motion_path),
        "motion_sha256": sha256(motion_path),
        "ready_reference": str(ready_path),
        "ready_reference_sha256": sha256(ready_path),
        "pd_pose": str(pose_path),
        "pd_pose_sha256": sha256(pose_path),
        "joint_order_version": T800_JOINT_ORDER_VERSION,
        "frames": int(joint_pos.shape[0]),
        "fps": float(motion["fps"]),
        "duration_s": float((joint_pos.shape[0] - 1) / float(motion["fps"])),
        "gates": gates,
        "thresholds": {
            "start_max_error_rad": args.start_max_error,
            "terminal_max_error_rad": args.terminal_max_error,
            "hold_max_drift_rad": args.hold_max_drift,
            "max_planar_displacement_m": args.max_planar_displacement,
        },
        "measurements": {
            "start_max_error_rad": float(start_error[worst_start_index]),
            "start_worst_joint": T800_POLICY_JOINT_NAMES[worst_start_index],
            "start_hold_max_drift_rad": start_drift,
            "terminal_max_error_rad": float(terminal_error[worst_terminal_index]),
            "terminal_worst_joint": T800_POLICY_JOINT_NAMES[worst_terminal_index],
            "terminal_hold_max_drift_rad": terminal_drift,
            "ready_reference_hold_max_drift_rad": ready_drift,
            "max_planar_displacement_m": displacement,
        },
        "simulation_required_gates": [
            "initial prone/supine base orientation",
            "floor contact set and impact",
            "head contact",
            "torque and joint limits",
            "stable upright boxing-ready hold",
        ],
        "errors": errors,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
