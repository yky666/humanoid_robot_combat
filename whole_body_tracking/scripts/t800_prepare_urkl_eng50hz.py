#!/usr/bin/env python3
"""Prepare URKL 50 Hz self-recorded T800 combat trajectories for training."""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

WBT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WBT_ROOT.parent
SOURCE_ROOT = WBT_ROOT / "source" / "whole_body_tracking"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from whole_body_tracking.robots.t800_joint_order import (  # noqa: E402
    T800_JOINT_ORDER_VERSION,
    T800_POLICY_JOINT_NAMES,
)


ACTION_LABELS = {
    "left_front_kick_002": {
        "action": "front_kick_left",
        "display_name": "left front kick",
        "deploy_slot_hint": "front_kick",
    },
    "left_hook_001": {
        "action": "hook_left",
        "display_name": "left hook",
        "deploy_slot_hint": "hook_left",
    },
    "rear_hook_001": {
        "action": "hook_rear",
        "display_name": "rear hook",
        "deploy_slot_hint": "hook_rear",
    },
    "roundhouse_kick_001": {
        "action": "roundhouse_kick",
        "display_name": "roundhouse kick",
        "deploy_slot_hint": "roundhouse_kick",
    },
    "straight_punch_L": {
        "action": "straight_punch_left",
        "display_name": "left straight punch",
        "deploy_slot_hint": "straight_punch_left",
    },
    "straight_punch_R": {
        "action": "straight_punch_right",
        "display_name": "right straight punch",
        "deploy_slot_hint": "straight_punch_right",
    },
}

REQUIRED_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "fps",
)


def scalar_fps(value: np.ndarray) -> float:
    fps = np.asarray(value).reshape(-1)
    if fps.size != 1:
        raise ValueError(f"fps should be scalar-like, got shape {np.asarray(value).shape}")
    return float(fps[0])


def finite_min_max(array: np.ndarray) -> tuple[float, float]:
    return float(np.nanmin(array)), float(np.nanmax(array))


def validate_motion(name: str, data: dict[str, np.ndarray]) -> list[str]:
    errors: list[str] = []
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        return [f"missing keys: {missing}"]

    frames = int(data["joint_pos"].shape[0])
    if data["joint_pos"].ndim != 2 or data["joint_pos"].shape[1] != len(T800_POLICY_JOINT_NAMES):
        errors.append(
            f"joint_pos must be (T, {len(T800_POLICY_JOINT_NAMES)}), got {data['joint_pos'].shape}"
        )
    if data["joint_vel"].shape != data["joint_pos"].shape:
        errors.append(f"joint_vel shape {data['joint_vel'].shape} does not match joint_pos {data['joint_pos'].shape}")
    for key, width in (
        ("body_pos_w", 3),
        ("body_quat_w", 4),
        ("body_lin_vel_w", 3),
        ("body_ang_vel_w", 3),
    ):
        if data[key].ndim != 3 or data[key].shape[0] != frames or data[key].shape[2] != width:
            errors.append(f"{key} should be (T, bodies, {width}), got {data[key].shape}")
    for key in REQUIRED_KEYS:
        arr = np.asarray(data[key])
        if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
            errors.append(f"{key} contains non-finite values")
    fps = scalar_fps(data["fps"])
    if fps <= 0 or not math.isfinite(fps):
        errors.append(f"invalid fps: {fps}")
    if frames < 20:
        errors.append(f"too few frames for training: {frames}")
    if errors:
        errors.insert(0, f"{name}: validation failed")
    return errors


def load_npz_from_zip(zip_file: zipfile.ZipFile, entry: str) -> dict[str, np.ndarray]:
    with zip_file.open(entry) as handle:
        raw = np.load(handle, allow_pickle=False)
        return {key: raw[key] for key in raw.files}


def stats_for_motion(name: str, data: dict[str, np.ndarray], output_path: Path, source_entry: str) -> dict:
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
    body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
    root_pos = body_pos[:, 0, :]
    fps = scalar_fps(data["fps"])
    root_delta = root_pos[-1, :2] - root_pos[0, :2]
    root_height_min = float(np.nanmin(root_pos[:, 2]))
    root_height_max = float(np.nanmax(root_pos[:, 2]))
    q_min, q_max = finite_min_max(joint_pos)
    qd_min, qd_max = finite_min_max(joint_vel)
    stem = Path(name).stem
    label = ACTION_LABELS.get(stem, {})
    return {
        "name": stem,
        "action": label.get("action", stem),
        "display_name": label.get("display_name", stem.replace("_", " ")),
        "deploy_slot_hint": label.get("deploy_slot_hint", stem),
        "smoke_run_name": f"smoke_urkl_eng50hz_{label.get('action', stem)}",
        "formal_run_name": f"urkl_eng50hz_{label.get('action', stem)}_t800_self_mimic_v1",
        "source_entry": source_entry,
        "source_kind": "urkl_eng_50hz_self_recorded",
        "output_path": str(output_path),
        "fps": fps,
        "frames": int(joint_pos.shape[0]),
        "duration_s": float(joint_pos.shape[0] / fps),
        "joint_dim": int(joint_pos.shape[1]),
        "body_count": int(body_pos.shape[1]),
        "root_height_min": root_height_min,
        "root_height_max": root_height_max,
        "root_xy_delta": [float(root_delta[0]), float(root_delta[1])],
        "joint_pos_min": q_min,
        "joint_pos_max": q_max,
        "joint_vel_min": qd_min,
        "joint_vel_max": qd_max,
        "competition_note": "Self-recorded URKL trajectory candidate; not derived from EngineAI official mimic.",
        "training_status": "prepared_not_trained",
    }


def write_motion_npz(data: dict[str, np.ndarray], output_path: Path, source_zip: Path, source_entry: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in data.items()}
    arrays["joint_names"] = np.asarray(T800_POLICY_JOINT_NAMES)
    arrays["joint_order_version"] = np.asarray(T800_JOINT_ORDER_VERSION)
    arrays["source_zip"] = np.asarray(str(source_zip))
    arrays["source_entry"] = np.asarray(source_entry)
    arrays["source_kind"] = np.asarray("urkl_eng_50hz_self_recorded")
    np.savez_compressed(output_path, **arrays)


def write_markdown(manifest: dict, output_path: Path) -> None:
    lines = [
        "# T800 URKL ENG 50Hz Prepared Trajectories",
        "",
        f"- Source zip: `{manifest['source_zip']}`",
        f"- Created at: `{manifest['created_at']}`",
        f"- Output root: `{manifest['output_root']}`",
        f"- Joint order: `{manifest['joint_order_version']}`",
        "",
        "| name | action | frames | seconds | bodies | root z | root xy delta | status |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for motion in manifest["motions"]:
        root_z = f"{motion['root_height_min']:.3f}..{motion['root_height_max']:.3f}"
        delta = f"{motion['root_xy_delta'][0]:.3f}, {motion['root_xy_delta'][1]:.3f}"
        lines.append(
            f"| `{motion['name']}` | {motion['action']} | {motion['frames']} | "
            f"{motion['duration_s']:.2f} | {motion['body_count']} | {root_z} | {delta} | "
            f"{motion['training_status']} |"
        )
    lines.extend(
        [
            "",
            "## Next Commands",
            "",
            "Use these NPZ files as competition-clean candidate references for short smoke training first.",
            "They are prepared assets only; no rollout success is implied until training and evaluation reports exist.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("/mnt/data/yangky/test/datasets/urkl_locomotion_260901/traj_eng_50hz.zip"),
        help="Path to traj_eng_50hz.zip.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "t800_urkl_eng50hz_20260910",
        help="Directory for prepared NPZ files and manifest.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing prepared NPZ files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_zip = args.zip.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)

    prepared_dir = output_root / "tracking_npz"
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_zip": str(source_zip),
        "output_root": str(output_root),
        "joint_order_version": T800_JOINT_ORDER_VERSION,
        "policy_joint_names": list(T800_POLICY_JOINT_NAMES),
        "motions": [],
    }

    with zipfile.ZipFile(source_zip) as zip_file:
        entries = sorted(name for name in zip_file.namelist() if name.endswith(".npz"))
        if not entries:
            raise RuntimeError(f"no npz entries found in {source_zip}")
        for entry in entries:
            data = load_npz_from_zip(zip_file, entry)
            errors = validate_motion(entry, data)
            if errors:
                raise RuntimeError("\n".join(errors))
            stem = Path(entry).stem
            output_path = prepared_dir / f"{stem}_tracking.npz"
            if output_path.exists() and not args.force:
                print(f"[SKIP] {output_path}")
            else:
                write_motion_npz(data, output_path, source_zip, entry)
                print(f"[OK] wrote {output_path}")
            manifest["motions"].append(stats_for_motion(entry, data, output_path, entry))

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(manifest, output_root / "README.md")
    print(f"[OK] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
