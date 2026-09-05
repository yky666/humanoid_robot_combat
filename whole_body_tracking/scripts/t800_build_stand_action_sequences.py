#!/usr/bin/env python3
"""Build ready-stance entry/action/exit GMR sequences for T800 joint training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


DEFAULT_WALKING_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "engineai_robotics_native_sdk/assets/config/t800/rl_walking_example/default.yaml"
)


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return quaternion / np.maximum(norm, 1.0e-8)


def as_wxyz(quaternion: np.ndarray, quaternion_format: str) -> np.ndarray:
    if quaternion_format == "wxyz":
        return quaternion
    if quaternion_format == "xyzw":
        return quaternion[..., [3, 0, 1, 2]]
    raise ValueError(f"Unsupported root quaternion format: {quaternion_format}")


def from_wxyz(quaternion: np.ndarray, quaternion_format: str) -> np.ndarray:
    if quaternion_format == "wxyz":
        return quaternion
    if quaternion_format == "xyzw":
        return quaternion[..., [1, 2, 3, 0]]
    raise ValueError(f"Unsupported root quaternion format: {quaternion_format}")


def yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = normalize_quaternion(np.asarray(quaternion, dtype=np.float64))
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_quaternion_wxyz(yaw: float) -> np.ndarray:
    return np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)


def quintic_phase(count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.float64)
    t = np.arange(1, count + 1, dtype=np.float64) / count
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def linear_blend(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    phase = quintic_phase(count)
    return start[None, ...] + phase.reshape((-1,) + (1,) * start.ndim) * (end - start)[None, ...]


def quaternion_blend(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    phase = quintic_phase(count)
    start = normalize_quaternion(np.asarray(start, dtype=np.float64))
    end = normalize_quaternion(np.asarray(end, dtype=np.float64))
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = start[None, :] + phase[:, None] * (end - start)[None, :]
        return normalize_quaternion(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    result = (
        np.sin((1.0 - phase) * theta)[:, None] / sin_theta * start[None, :]
        + np.sin(phase * theta)[:, None] / sin_theta * end[None, :]
    )
    return normalize_quaternion(result)


def load_raw_motion(path: Path) -> dict[str, np.ndarray | float | str]:
    with np.load(path, allow_pickle=False) as data:
        quaternion_format = str(np.asarray(data.get("root_rot_format", "wxyz")).reshape(-1)[0]).lower()
        return {
            "fps": float(np.asarray(data["fps"]).reshape(-1)[0]),
            "root_pos": np.asarray(data["root_pos"], dtype=np.float64),
            "root_rot": as_wxyz(np.asarray(data["root_rot"], dtype=np.float64), quaternion_format),
            "dof_pos": np.asarray(data["dof_pos"], dtype=np.float64),
            "root_rot_format": quaternion_format,
        }


def resolve(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def load_walking_joint_pos(path: Path) -> np.ndarray:
    config_path = path.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    joint_pos = np.asarray(
        [value for group in config["default_joint_q"] for value in group], dtype=np.float64
    )
    if joint_pos.shape != (25,):
        raise ValueError(f"Expected 25 walking joints in {config_path}, got {joint_pos.shape}")
    return joint_pos


def build_sequence(
    action: dict[str, np.ndarray | float | str],
    ready_joint_pos: np.ndarray,
    ready_height: float,
    hold_frames: int,
    blend_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_root_pos = np.asarray(action["root_pos"])
    action_root_rot = np.asarray(action["root_rot"])
    action_joint_pos = np.asarray(action["dof_pos"])

    start_ready_pos = action_root_pos[0].copy()
    start_ready_pos[2] = ready_height
    end_ready_pos = action_root_pos[-1].copy()
    end_ready_pos[2] = ready_height
    start_ready_rot = yaw_quaternion_wxyz(yaw_from_wxyz(action_root_rot[0]))
    end_ready_rot = yaw_quaternion_wxyz(yaw_from_wxyz(action_root_rot[-1]))

    root_pos_parts = [
        np.repeat(start_ready_pos[None, :], hold_frames, axis=0),
        linear_blend(start_ready_pos, action_root_pos[0], blend_frames),
        action_root_pos[1:],
        linear_blend(action_root_pos[-1], end_ready_pos, blend_frames),
        np.repeat(end_ready_pos[None, :], hold_frames, axis=0),
    ]
    root_rot_parts = [
        np.repeat(start_ready_rot[None, :], hold_frames, axis=0),
        quaternion_blend(start_ready_rot, action_root_rot[0], blend_frames),
        action_root_rot[1:],
        quaternion_blend(action_root_rot[-1], end_ready_rot, blend_frames),
        np.repeat(end_ready_rot[None, :], hold_frames, axis=0),
    ]
    joint_pos_parts = [
        np.repeat(ready_joint_pos[None, :], hold_frames, axis=0),
        linear_blend(ready_joint_pos, action_joint_pos[0], blend_frames),
        action_joint_pos[1:],
        linear_blend(action_joint_pos[-1], ready_joint_pos, blend_frames),
        np.repeat(ready_joint_pos[None, :], hold_frames, axis=0),
    ]
    return (
        np.concatenate(root_pos_parts, axis=0).astype(np.float32),
        normalize_quaternion(np.concatenate(root_rot_parts, axis=0)).astype(np.float32),
        np.concatenate(joint_pos_parts, axis=0).astype(np.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracking-output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--walking-config", type=Path, default=DEFAULT_WALKING_CONFIG)
    parser.add_argument("--walking-root-height", type=float, default=0.8)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--blend-seconds", type=float, default=1.0)
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=9.0,
        help="Pad every sequence with its final ready pose to a common duration.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    motions = manifest["motions"]
    walking_config = args.walking_config.expanduser().resolve()
    ready_joint_pos = load_walking_joint_pos(walking_config)
    ready_height = float(args.walking_root_height)

    output_dir = args.output_dir.expanduser().resolve()
    tracking_output_dir = args.tracking_output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_output_dir.mkdir(parents=True, exist_ok=True)

    output_motions = []
    for entry in motions:
        if entry.get("group") != "standing_action":
            continue
        input_path = resolve(repo_root, entry["raw_gmr"])
        action = load_raw_motion(input_path)
        fps = float(action["fps"])
        hold_frames = max(1, round(args.hold_seconds * fps))
        blend_frames = max(1, round(args.blend_seconds * fps))
        root_pos, root_rot_wxyz, dof_pos = build_sequence(
            action, ready_joint_pos, ready_height, hold_frames, blend_frames
        )
        target_frames = round(args.target_seconds * fps)
        if root_pos.shape[0] > target_frames:
            raise ValueError(
                f"{entry['name']} needs {root_pos.shape[0]} frames but target duration only allows {target_frames}."
            )
        pad_frames = target_frames - root_pos.shape[0]
        if pad_frames:
            root_pos = np.concatenate([root_pos, np.repeat(root_pos[-1:], pad_frames, axis=0)], axis=0)
            root_rot_wxyz = np.concatenate(
                [root_rot_wxyz, np.repeat(root_rot_wxyz[-1:], pad_frames, axis=0)], axis=0
            )
            dof_pos = np.concatenate([dof_pos, np.repeat(dof_pos[-1:], pad_frames, axis=0)], axis=0)

        output_path = output_dir / f"{entry['name']}_stand_action_stand.npz"
        tracking_path = tracking_output_dir / f"{entry['name']}_stand_action_stand_tracking.npz"
        quaternion_format = str(action["root_rot_format"])
        np.savez(
            output_path,
            fps=np.float32(fps),
            root_pos=root_pos,
            root_rot=from_wxyz(root_rot_wxyz, quaternion_format).astype(np.float32),
            dof_pos=dof_pos,
            root_rot_format=np.asarray(quaternion_format),
            sequence_layout=np.asarray(
                "walking_zero_hold,entry_blend,action,exit_blend,walking_zero_hold"
            ),
            ready_source=np.asarray(str(walking_config)),
        )
        output_motions.append(
            {
                "name": entry["name"],
                "group": "standing_transition",
                "official_target": entry.get("official_target", entry["name"]),
                "raw_gmr": str(output_path),
                "motion_file": str(tracking_path),
                "frames": int(root_pos.shape[0]),
                "fps": fps,
            }
        )
        print(
            f"[OK] {entry['name']}: {output_path} frames={root_pos.shape[0]} "
            f"duration={root_pos.shape[0] / fps:.2f}s"
        )

    output_manifest = args.output_manifest.expanduser().resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(
            {
                "description": "T800 walking-zero entry/action/exit references for gated joint training.",
                "joint_order_version": manifest.get("joint_order_version", "t800_policy_v1"),
                "ready_source": str(walking_config),
                "walking_root_height": ready_height,
                "motions": output_motions,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[OK] manifest: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
