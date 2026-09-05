#!/usr/bin/env python3
"""Compose T800 fall-recovery reference fragments into full recovery candidates."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


DEFAULT_WALKING_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "engineai_robotics_native_sdk/assets/config/t800/rl_walking_example/default.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data2/yangky/test/whole_body_tracking/artifacts/required_motion_T800/gmr_pkl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/yangky/test/whole_body_tracking/artifacts/recovery_chain_T800/gmr_pkl"),
    )
    parser.add_argument("--transition", default="transition_male2_crouch_to_ready_stageii.pkl")
    parser.add_argument("--start-hold-seconds", type=float, default=0.5)
    parser.add_argument("--blend-seconds", type=float, default=0.45)
    parser.add_argument("--walking-blend-seconds", type=float, default=1.0)
    parser.add_argument("--end-hold-seconds", type=float, default=1.2)
    parser.add_argument("--walking-config", type=Path, default=DEFAULT_WALKING_CONFIG)
    parser.add_argument("--walking-root-height", type=float, default=0.8)
    parser.add_argument(
        "--npz-output-root",
        type=Path,
        default=None,
        help="Optionally write the same portable raw motion as NPZ.",
    )
    parser.add_argument(
        "--base-motion",
        type=Path,
        default=None,
        help="Preserve an approved PKL motion and only append its walking-zero endpoint.",
    )
    parser.add_argument(
        "--transition-search-frames",
        type=int,
        default=0,
        help="Search this many leading transition frames for the closest crouch splice pose. 0 keeps frame 0.",
    )
    parser.add_argument(
        "--blend-mode",
        choices=("insert", "overlap"),
        default="insert",
        help="insert preserves the original bridge. overlap crossfades the tail of the first clip with transition.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Compose only chain names containing this substring. Can be passed multiple times.",
    )
    return parser.parse_args()


def load_motion(path: Path) -> dict:
    with path.open("rb") as f:
        motion = pickle.load(f)
    return {
        **motion,
        "fps": float(motion["fps"]),
        "root_pos": np.asarray(motion["root_pos"], dtype=np.float64),
        "root_rot": normalize_quat(np.asarray(motion["root_rot"], dtype=np.float64)),
        "dof_pos": np.asarray(motion["dof_pos"], dtype=np.float64),
    }


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return quat / norm


def load_walking_joint_pos(path: Path) -> np.ndarray:
    config_path = path.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    joint_pos = np.asarray(
        [value for group in config["default_joint_q"] for value in group], dtype=np.float64
    )
    if joint_pos.shape != (25,):
        raise ValueError(f"Expected 25 walking joints in {config_path}, got {joint_pos.shape}")
    return joint_pos


def quintic_phase(count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.float64)
    t = np.arange(1, count + 1, dtype=np.float64) / count
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def quat_inverse_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    out = np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )
    return normalize_quat(out)


def yaw_from_quat_xyzw(q: np.ndarray) -> float:
    mat = R.from_quat(q).as_matrix()
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def yaw_quat_xyzw(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float64)


def reverse_motion(motion: dict) -> dict:
    return {
        **motion,
        "root_pos": motion["root_pos"][::-1].copy(),
        "root_rot": motion["root_rot"][::-1].copy(),
        "dof_pos": motion["dof_pos"][::-1].copy(),
    }


def align_second_to_first(first: dict, second: dict, anchor_index: int = 0) -> dict:
    anchor_index = int(np.clip(anchor_index, 0, len(second["root_pos"]) - 1))
    delta_q = yaw_quat_xyzw(
        yaw_from_quat_xyzw(first["root_rot"][-1]) - yaw_from_quat_xyzw(second["root_rot"][anchor_index])
    )
    delta_rot = R.from_quat(delta_q)
    rel_pos = second["root_pos"] - second["root_pos"][anchor_index]
    aligned_root_pos = delta_rot.apply(rel_pos) + first["root_pos"][-1]
    aligned_root_rot = quat_mul_xyzw(np.broadcast_to(delta_q, second["root_rot"].shape), second["root_rot"])
    return {
        **second,
        "root_pos": aligned_root_pos,
        "root_rot": aligned_root_rot,
    }


def make_hold(motion: dict, frame_index: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count <= 0:
        return motion["root_pos"][:0], motion["root_rot"][:0], motion["dof_pos"][:0]
    root_pos = np.repeat(motion["root_pos"][frame_index][None, :], count, axis=0)
    root_rot = np.repeat(motion["root_rot"][frame_index][None, :], count, axis=0)
    dof_pos = np.repeat(motion["dof_pos"][frame_index][None, :], count, axis=0)
    return root_pos, root_rot, dof_pos


def make_blend(first: dict, second: dict, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count <= 0:
        return first["root_pos"][:0], first["root_rot"][:0], first["dof_pos"][:0]
    alpha = np.linspace(0.0, 1.0, count + 2, dtype=np.float64)[1:-1]
    root_pos = (1.0 - alpha[:, None]) * first["root_pos"][-1] + alpha[:, None] * second["root_pos"][0]
    q0 = first["root_rot"][-1]
    q1 = second["root_rot"][0]
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    root_rot = normalize_quat((1.0 - alpha[:, None]) * q0 + alpha[:, None] * q1)
    dof_pos = (1.0 - alpha[:, None]) * first["dof_pos"][-1] + alpha[:, None] * second["dof_pos"][0]
    return root_pos, root_rot, dof_pos


def make_walking_blend(
    motion: dict,
    walking_joint_pos: np.ndarray,
    walking_root_height: float,
    count: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    target_root_pos = motion["root_pos"][-1].copy()
    target_root_pos[2] = walking_root_height
    target_root_rot = yaw_quat_xyzw(yaw_from_quat_xyzw(motion["root_rot"][-1]))
    phase = quintic_phase(count)
    root_pos = motion["root_pos"][-1][None, :] + phase[:, None] * (
        target_root_pos - motion["root_pos"][-1]
    )[None, :]

    start_rot = motion["root_rot"][-1]
    if float(np.dot(start_rot, target_root_rot)) < 0.0:
        target_root_rot = -target_root_rot
    root_rot = normalize_quat(start_rot[None, :] + phase[:, None] * (target_root_rot - start_rot)[None, :])
    dof_pos = motion["dof_pos"][-1][None, :] + phase[:, None] * (
        walking_joint_pos - motion["dof_pos"][-1]
    )[None, :]
    target = {
        **motion,
        "root_pos": target_root_pos[None, :],
        "root_rot": target_root_rot[None, :],
        "dof_pos": walking_joint_pos[None, :],
    }
    return (root_pos, root_rot, dof_pos), target


def find_transition_anchor(first: dict, transition: dict, search_frames: int) -> tuple[int, float]:
    if search_frames <= 0:
        return 0, float("nan")

    limit = min(int(search_frames), len(transition["dof_pos"]))
    first_dof = first["dof_pos"][-1]
    first_z = float(first["root_pos"][-1, 2])
    best_index = 0
    best_score = float("inf")
    for index in range(limit):
        dof_rms = float(np.linalg.norm(transition["dof_pos"][index] - first_dof) / np.sqrt(first_dof.size))
        z_error = abs(float(transition["root_pos"][index, 2]) - first_z)
        score = dof_rms + 2.0 * z_error
        if score < best_score:
            best_score = score
            best_index = index
    return best_index, best_score


def make_overlap_blend(first: dict, second: dict, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    count = min(count, len(first["root_pos"]), len(second["root_pos"]))
    if count <= 0:
        return first["root_pos"][:0], first["root_rot"][:0], first["dof_pos"][:0], 0

    alpha = np.linspace(0.0, 1.0, count, dtype=np.float64)
    first_pos = first["root_pos"][-count:]
    first_rot = first["root_rot"][-count:]
    first_dof = first["dof_pos"][-count:]
    second_pos = second["root_pos"][:count]
    second_rot = second["root_rot"][:count].copy()
    second_dof = second["dof_pos"][:count]

    flip_mask = np.sum(first_rot * second_rot, axis=1) < 0.0
    second_rot[flip_mask] *= -1.0
    root_pos = (1.0 - alpha[:, None]) * first_pos + alpha[:, None] * second_pos
    root_rot = normalize_quat((1.0 - alpha[:, None]) * first_rot + alpha[:, None] * second_rot)
    dof_pos = (1.0 - alpha[:, None]) * first_dof + alpha[:, None] * second_dof
    return root_pos, root_rot, dof_pos, count


def write_motion(name: str, out: dict, args: argparse.Namespace, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{name}.pkl"
    with output_path.open("wb") as f:
        pickle.dump(out, f)
    if args.npz_output_root is not None:
        npz_output_root = args.npz_output_root.expanduser().resolve()
        npz_output_root.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_output_root / f"{name}.npz",
            fps=np.float32(out["fps"]),
            root_pos=out["root_pos"],
            root_rot=out["root_rot"],
            dof_pos=out["dof_pos"],
            root_rot_format=np.asarray("xyzw"),
        )
    print(
        f"[OK] {name}: frames={len(out['root_pos'])} fps={out['fps']:.3f} "
        f"duration={len(out['root_pos']) / out['fps']:.3f}s -> {output_path}"
    )
    return output_path


def append_walking_endpoint(
    name: str,
    motion: dict,
    walking_joint_pos: np.ndarray,
    args: argparse.Namespace,
    output_root: Path,
) -> Path:
    fps = float(motion["fps"])
    walking_blend_count = int(round(args.walking_blend_seconds * fps))
    end_hold = int(round(args.end_hold_seconds * fps))
    walking_blend, walking_target = make_walking_blend(
        motion, walking_joint_pos, args.walking_root_height, walking_blend_count
    )
    hold = make_hold(walking_target, 0, end_hold)
    root_pos = np.concatenate([motion["root_pos"], walking_blend[0], hold[0]], axis=0).astype(np.float32)
    root_rot = np.concatenate([motion["root_rot"], walking_blend[1], hold[1]], axis=0).astype(np.float32)
    dof_pos = np.concatenate([motion["dof_pos"], walking_blend[2], hold[2]], axis=0).astype(np.float32)
    out = {
        **motion,
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "root_rot_format": "xyzw",
        "walking_config": str(args.walking_config.expanduser().resolve()),
        "walking_root_height": float(args.walking_root_height),
        "walking_blend_seconds": float(args.walking_blend_seconds),
        "endpoint": "engineai_official_walking_zero_command",
        "compose_notes": (
            f"{motion.get('compose_notes', '')} Preserved the approved motion prefix and appended a smooth "
            "transition to the EngineAI walking policy zero-command pose."
        ).strip(),
    }
    return write_motion(name, out, args, output_root)


def compose(
    name: str,
    first: dict,
    transition: dict,
    walking_joint_pos: np.ndarray,
    args: argparse.Namespace,
    output_root: Path,
) -> Path:
    anchor_index, anchor_score = find_transition_anchor(first, transition, args.transition_search_frames)
    transition = align_second_to_first(first, transition, anchor_index=anchor_index)
    if anchor_index > 0:
        transition = {
            **transition,
            "root_pos": transition["root_pos"][anchor_index:].copy(),
            "root_rot": transition["root_rot"][anchor_index:].copy(),
            "dof_pos": transition["dof_pos"][anchor_index:].copy(),
        }

    fps = first["fps"]
    start_hold = int(round(args.start_hold_seconds * fps))
    blend_count = int(round(args.blend_seconds * fps))
    walking_blend_count = int(round(args.walking_blend_seconds * fps))
    end_hold = int(round(args.end_hold_seconds * fps))

    parts = []
    parts.append(make_hold(first, 0, start_hold))
    if args.blend_mode == "overlap":
        overlap_pos, overlap_rot, overlap_dof, used_blend = make_overlap_blend(first, transition, blend_count)
        parts.append((first["root_pos"][:-used_blend], first["root_rot"][:-used_blend], first["dof_pos"][:-used_blend]))
        parts.append((overlap_pos, overlap_rot, overlap_dof))
        parts.append((transition["root_pos"][used_blend:], transition["root_rot"][used_blend:], transition["dof_pos"][used_blend:]))
    else:
        parts.append((first["root_pos"], first["root_rot"], first["dof_pos"]))
        parts.append(make_blend(first, transition, blend_count))
        parts.append((transition["root_pos"][1:], transition["root_rot"][1:], transition["dof_pos"][1:]))
    walking_blend, walking_target = make_walking_blend(
        transition, walking_joint_pos, args.walking_root_height, walking_blend_count
    )
    parts.append(walking_blend)
    parts.append(make_hold(walking_target, 0, end_hold))

    root_pos = np.concatenate([p[0] for p in parts], axis=0).astype(np.float32)
    root_rot = normalize_quat(np.concatenate([p[1] for p in parts], axis=0)).astype(np.float32)
    dof_pos = np.concatenate([p[2] for p in parts], axis=0).astype(np.float32)

    out = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "root_rot_format": "xyzw",
        "source_fragments": [first.get("source_name", "first"), transition.get("source_name", "transition")],
        "splice_anchor_index": anchor_index,
        "splice_anchor_score": anchor_score,
        "blend_mode": args.blend_mode,
        "walking_config": str(args.walking_config.expanduser().resolve()),
        "walking_root_height": float(args.walking_root_height),
        "walking_blend_seconds": float(args.walking_blend_seconds),
        "endpoint": "engineai_official_walking_zero_command",
        "compose_notes": (
            "Training-audit candidate composed from a floor-to-crouch fragment, a crouch-to-ready fragment, "
            "and a smooth transition to the EngineAI walking policy zero-command pose. "
            "Use only after visual approval."
        ),
    }
    return write_motion(name, out, args, output_root)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    walking_joint_pos = load_walking_joint_pos(args.walking_config)
    if args.base_motion is not None:
        base_path = args.base_motion.expanduser().resolve()
        with base_path.open("rb") as f:
            motion = pickle.load(f)
        quaternion_format = str(motion.get("root_rot_format", "xyzw")).lower()
        if quaternion_format != "xyzw":
            raise ValueError(f"Expected xyzw root rotations in {base_path}, got {quaternion_format!r}")
        motion = {
            **motion,
            "fps": float(motion["fps"]),
            "root_pos": np.asarray(motion["root_pos"], dtype=np.float64),
            "root_rot": np.asarray(motion["root_rot"], dtype=np.float64),
            "dof_pos": np.asarray(motion["dof_pos"], dtype=np.float64),
            "base_motion": str(base_path),
        }
        append_walking_endpoint(base_path.stem, motion, walking_joint_pos, args, output_root)
        return 0

    transition = load_motion(input_root / args.transition)
    transition["source_name"] = args.transition

    chain_specs = [
        ("recovery_supine_male1_to_ready", "getup_male1_supine_to_crouch_stageii.pkl", False),
        ("recovery_supine_male2_to_ready", "getup_male2_supine_to_crouch_stageii.pkl", False),
        ("recovery_supine_female1_to_ready", "getup_female1_supine_to_crouch_stageii.pkl", False),
        ("recovery_prone_male1_to_ready", "fall_male1_crouch_to_prone_stageii.pkl", True),
        ("recovery_supine_male1_reverse_fall_to_ready", "fall_male1_crouch_to_supine_stageii.pkl", True),
        ("recovery_supine_male2_reverse_fall_to_ready", "fall_male2_crouch_to_supine_stageii.pkl", True),
    ]

    for name, fragment_name, reverse in chain_specs:
        fragment = load_motion(input_root / fragment_name)
        fragment["source_name"] = fragment_name
        if reverse:
            fragment = reverse_motion(fragment)
            fragment["source_name"] = f"reverse({fragment_name})"
        if args.only and not any(pattern in name for pattern in args.only):
            continue
        compose(name, fragment, transition, walking_joint_pos, args, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
