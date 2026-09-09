#!/usr/bin/env python3
"""Extract a measured T800 boxing-ready pose from EngineAI SDK joint logs."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np


WBT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WBT_ROOT / "source" / "whole_body_tracking"))

from whole_body_tracking.robots.t800_joint_order import (  # noqa: E402
    T800_JOINT_ORDER_VERSION,
    T800_POLICY_JOINT_NAMES,
    T800_SDK_POLICY_JOINT_NAMES,
)


DEFAULT_ARCHIVE = Path("/mnt/data/yangky/test/datasets/urkl_locomotion_260901/motion_logs.zip")
DEFAULT_MOTION = "logs/pdstand2baoquan.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ARCHIVE, help="CSV file or zip archive.")
    parser.add_argument("--member", default=DEFAULT_MOTION, help="CSV member when --input is a zip archive.")
    parser.add_argument("--tail-seconds", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument(
        "--output-tail-npz",
        type=Path,
        default=None,
        help="Optional tail reference motion NPZ with joint_pos/joint_vel/joint_tau in policy order.",
    )
    return parser.parse_args()


def read_csv_text(path: Path, member: str) -> str:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as handle:
                return handle.read().decode("utf-8-sig")
    return path.read_text(encoding="utf-8-sig")


def load_joint_log(path: Path, member: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    reader = csv.reader(io.StringIO(read_csv_text(path, member)))
    header = next(reader)
    rows = [[float(value) for value in row] for row in reader if row]
    if not rows:
        raise ValueError(f"{path}:{member} contains no samples")
    data = np.asarray(rows, dtype=np.float64)
    columns = {name: index for index, name in enumerate(header)}

    missing = []
    sdk_pos_names = []
    sdk_vel_names = []
    sdk_tau_names = []
    for joint_name in T800_SDK_POLICY_JOINT_NAMES:
        sdk_pos_names.append(f"pos_{joint_name}")
        sdk_vel_names.append(f"vel_{joint_name}")
        sdk_tau_names.append(f"tau_{joint_name}")
        for column_name in (sdk_pos_names[-1], sdk_vel_names[-1], sdk_tau_names[-1]):
            if column_name not in columns:
                missing.append(column_name)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    time_s = data[:, columns["t_host"]]
    joint_pos = data[:, [columns[name] for name in sdk_pos_names]]
    joint_vel = data[:, [columns[name] for name in sdk_vel_names]]
    joint_tau = data[:, [columns[name] for name in sdk_tau_names]]
    return time_s, joint_pos, joint_vel, joint_tau, T800_SDK_POLICY_JOINT_NAMES


def sampling_rate(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def summarize_tail(values: np.ndarray, fps: float, tail_seconds: float) -> tuple[np.ndarray, np.ndarray, int]:
    if tail_seconds <= 0:
        raise ValueError("--tail-seconds must be positive")
    tail_frames = max(1, min(len(values), int(round(tail_seconds * fps)))) if fps > 0 else len(values)
    tail = values[-tail_frames:]
    return np.median(tail, axis=0), np.std(tail, axis=0), tail_frames


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_npz = args.output_npz.expanduser().resolve() if args.output_npz else None
    output_tail_npz = args.output_tail_npz.expanduser().resolve() if args.output_tail_npz else None

    time_s, joint_pos, joint_vel, joint_tau, sdk_names = load_joint_log(input_path, args.member)
    fps = sampling_rate(time_s)
    target_pos, tail_pos_std, tail_frames = summarize_tail(joint_pos, fps, args.tail_seconds)
    target_vel, tail_vel_std, _ = summarize_tail(joint_vel, fps, args.tail_seconds)
    target_tau, tail_tau_std, _ = summarize_tail(joint_tau, fps, args.tail_seconds)

    target_by_policy = {
        policy_name: float(value)
        for policy_name, value in zip(T800_POLICY_JOINT_NAMES, target_pos, strict=True)
    }
    report = {
        "source": str(input_path),
        "member": args.member,
        "joint_order_version": T800_JOINT_ORDER_VERSION,
        "source_joint_names": sdk_names,
        "policy_joint_names": T800_POLICY_JOINT_NAMES,
        "frame_count": int(len(time_s)),
        "duration_s": float(time_s[-1] - time_s[0]),
        "estimated_fps": fps,
        "tail_seconds": float(args.tail_seconds),
        "tail_frames": int(tail_frames),
        "target_joint_pos": [float(value) for value in target_pos],
        "target_joint_vel_median": [float(value) for value in target_vel],
        "target_joint_tau_median": [float(value) for value in target_tau],
        "tail_joint_pos_std": [float(value) for value in tail_pos_std],
        "tail_joint_vel_std": [float(value) for value in tail_vel_std],
        "tail_joint_tau_std": [float(value) for value in tail_tau_std],
        "target_by_policy_joint": target_by_policy,
        "max_tail_joint_pos_std": float(np.max(tail_pos_std)),
        "mean_tail_joint_pos_std": float(np.mean(tail_pos_std)),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if output_npz is not None:
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            joint_order_version=np.asarray(T800_JOINT_ORDER_VERSION),
            joint_names=np.asarray(T800_POLICY_JOINT_NAMES),
            source_joint_names=np.asarray(sdk_names),
            target_joint_pos=target_pos.astype(np.float32),
            target_joint_vel_median=target_vel.astype(np.float32),
            target_joint_tau_median=target_tau.astype(np.float32),
            tail_joint_pos_std=tail_pos_std.astype(np.float32),
            tail_joint_vel_std=tail_vel_std.astype(np.float32),
            tail_joint_tau_std=tail_tau_std.astype(np.float32),
            estimated_fps=np.asarray(fps, dtype=np.float32),
            tail_frames=np.asarray(tail_frames, dtype=np.int32),
        )

    if output_tail_npz is not None:
        tail_slice = slice(len(time_s) - tail_frames, len(time_s))
        output_tail_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_tail_npz,
            reference_type=np.asarray("real_t800_baoquan_tail_joint_log"),
            joint_order_version=np.asarray(T800_JOINT_ORDER_VERSION),
            joint_names=np.asarray(T800_POLICY_JOINT_NAMES),
            source_joint_names=np.asarray(sdk_names),
            source=np.asarray(str(input_path)),
            member=np.asarray(args.member),
            fps=np.asarray(fps, dtype=np.float32),
            time_s=(time_s[tail_slice] - time_s[tail_slice][0]).astype(np.float32),
            joint_pos=joint_pos[tail_slice].astype(np.float32),
            joint_vel=joint_vel[tail_slice].astype(np.float32),
            joint_tau=joint_tau[tail_slice].astype(np.float32),
            target_joint_pos=target_pos.astype(np.float32),
        )

    print(f"[OK] wrote {output_json}")
    if output_npz is not None:
        print(f"[OK] wrote {output_npz}")
    if output_tail_npz is not None:
        print(f"[OK] wrote {output_tail_npz}")
    print("[INFO] target_joint_pos =", json.dumps(report["target_joint_pos"]))
    print(f"[INFO] tail std max={report['max_tail_joint_pos_std']:.6f}, mean={report['mean_tail_joint_pos_std']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
