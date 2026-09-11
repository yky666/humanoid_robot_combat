#!/usr/bin/env python3
"""Audit URKL locomotion trajectories for baoquan command-conditioned training."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

WBT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WBT_ROOT.parent
SOURCE_ROOT = WBT_ROOT / "source" / "whole_body_tracking"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from whole_body_tracking.robots.t800_joint_order import T800_POLICY_JOINT_NAMES  # noqa: E402


LABEL_TO_COMMAND_HINT = {
    "前进": "forward",
    "后退": "backward",
    "左移": "left_strafe",
    "右移": "right_strafe",
}

UPPER_BODY_NAMES = {
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J20_SHOULDER_PITCH_R",
    "J21_SHOULDER_ROLL_R",
    "J22_SHOULDER_YAW_R",
    "J23_ELBOW_PITCH_R",
    "J24_ELBOW_YAW_R",
    "J27_HEAD_PITCH",
    "J28_HEAD_YAW",
}
UPPER_BODY_INDICES = [i for i, name in enumerate(T800_POLICY_JOINT_NAMES) if name in UPPER_BODY_NAMES]


def scalar_fps(value: np.ndarray) -> float:
    fps = np.asarray(value).reshape(-1)
    if fps.size != 1:
        raise ValueError(f"fps should be scalar-like, got shape {np.asarray(value).shape}")
    return float(fps[0])


def load_baoquan_target(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "joint_pos" not in data:
            return None
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        if joint_pos.ndim == 2:
            return joint_pos[-1]
        if joint_pos.ndim == 1:
            return joint_pos
    return None


def dominant_axis(delta_xy: np.ndarray) -> str:
    if abs(float(delta_xy[0])) >= abs(float(delta_xy[1])):
        return "+x" if delta_xy[0] >= 0 else "-x"
    return "+y" if delta_xy[1] >= 0 else "-y"


def audit_file(path: Path, root: Path, baoquan_target: np.ndarray | None) -> dict:
    with np.load(path, allow_pickle=False) as data:
        fps = scalar_fps(data["fps"])
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
    root_pos = body_pos[:, 0, :]
    duration = float(joint_pos.shape[0] / fps)
    delta_xy = root_pos[-1, :2] - root_pos[0, :2]
    distance_xy = float(np.linalg.norm(delta_xy))
    mean_velocity_xy = delta_xy / max(duration, 1e-6)
    root_z_min = float(np.nanmin(root_pos[:, 2]))
    root_z_max = float(np.nanmax(root_pos[:, 2]))
    label = path.parent.name
    command_hint = LABEL_TO_COMMAND_HINT.get(label, label)
    has_motion = distance_xy >= 0.5 and float(np.linalg.norm(mean_velocity_xy)) >= 0.05
    height_ok = root_z_min >= 0.65
    quality = "usable"
    reasons: list[str] = []
    if not has_motion:
        quality = "reject"
        reasons.append("insufficient root displacement for locomotion command labeling")
    if not height_ok:
        quality = "review" if quality == "usable" else quality
        reasons.append("low root height segment; may contain crouch/fall/noisy frames")

    baoquan_upper_l2 = None
    if baoquan_target is not None and baoquan_target.shape[0] == joint_pos.shape[1]:
        mean_upper = np.mean(joint_pos[:, UPPER_BODY_INDICES], axis=0)
        target_upper = baoquan_target[UPPER_BODY_INDICES]
        baoquan_upper_l2 = float(np.linalg.norm(mean_upper - target_upper) / max(len(UPPER_BODY_INDICES), 1) ** 0.5)

    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "label": label,
        "command_hint": command_hint,
        "frames": int(joint_pos.shape[0]),
        "fps": fps,
        "duration_s": duration,
        "joint_dim": int(joint_pos.shape[1]),
        "body_count": int(body_pos.shape[1]),
        "root_xy_delta": [float(delta_xy[0]), float(delta_xy[1])],
        "root_xy_distance": distance_xy,
        "mean_velocity_xy": [float(mean_velocity_xy[0]), float(mean_velocity_xy[1])],
        "dominant_world_axis": dominant_axis(delta_xy),
        "root_height_min": root_z_min,
        "root_height_max": root_z_max,
        "joint_vel_abs_max": float(np.nanmax(np.abs(joint_vel))),
        "baoquan_upper_l2": baoquan_upper_l2,
        "quality": quality,
        "reasons": reasons,
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# T800 Baoquan Locomotion Dataset Audit",
        "",
        f"- Created at: `{report['created_at']}`",
        f"- Dataset root: `{report['dataset_root']}`",
        f"- Baoquan target: `{report.get('baoquan_target') or 'not used'}`",
        "",
        "| file | label | command | seconds | root xy delta | mean vxy | axis | z range | baoquan upper L2 | quality |",
        "|---|---|---|---:|---|---|---|---|---:|---|",
    ]
    for item in report["files"]:
        delta = f"{item['root_xy_delta'][0]:.3f}, {item['root_xy_delta'][1]:.3f}"
        vxy = f"{item['mean_velocity_xy'][0]:.3f}, {item['mean_velocity_xy'][1]:.3f}"
        z_range = f"{item['root_height_min']:.3f}..{item['root_height_max']:.3f}"
        l2 = "" if item["baoquan_upper_l2"] is None else f"{item['baoquan_upper_l2']:.3f}"
        lines.append(
            f"| `{item['relative_path']}` | {item['label']} | {item['command_hint']} | "
            f"{item['duration_s']:.2f} | {delta} | {vxy} | {item['dominant_world_axis']} | "
            f"{z_range} | {l2} | {item['quality']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `usable` means the trajectory has enough root displacement to estimate a command label.",
            "- `review` means the trajectory moves but contains low-root-height segments or other safety caveats.",
            "- `reject` means it should not drive velocity-command training without trimming or relabeling.",
            "",
            "For the next locomotion training pass, keep upper-body baoquan as a target and use these movement files only as lower-body style/command data.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/data/yangky/test/datasets/urkl_locomotion_260901/tracking_npz"),
        help="Root containing 前进/后退/左移/右移 tracking npz files.",
    )
    parser.add_argument(
        "--baoquan-target",
        type=Path,
        default=PROJECT_ROOT / "results" / "t800_real_baoquan_getup_20260907" / "baoquan_tail_reference_2s.npz",
        help="Optional measured baoquan target NPZ used for upper-body distance.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "t800_baoquan_locomotion_20260910",
        help="Directory for audit outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    baoquan_target_path = args.baoquan_target.expanduser().resolve()
    baoquan_target = load_baoquan_target(baoquan_target_path)
    files = sorted(dataset_root.glob("*/*.npz"))
    if not files:
        raise RuntimeError(f"no npz files found under {dataset_root}")
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "baoquan_target": str(baoquan_target_path) if baoquan_target is not None else None,
        "upper_body_joint_names": [T800_POLICY_JOINT_NAMES[i] for i in UPPER_BODY_INDICES],
        "files": [audit_file(path, dataset_root, baoquan_target) for path in files],
    }
    summary = {
        "total": len(report["files"]),
        "usable": sum(1 for item in report["files"] if item["quality"] == "usable"),
        "review": sum(1 for item in report["files"] if item["quality"] == "review"),
        "reject": sum(1 for item in report["files"] if item["quality"] == "reject"),
    }
    report["summary"] = summary
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "dataset_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, output_root / "README.md")
    print(f"[OK] wrote {report_path}")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
