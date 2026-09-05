#!/usr/bin/env python3
"""Convert MotionDecode Unitree-G1 CSV samples to T800 motion files.

The MotionDecode ``samples`` release stores retargeted Unitree G1 trajectories
as CSV files with root pose plus 29 joint columns.  This bridge keeps the root
trajectory, maps semantically matching joints into the 25-DoF T800 order used
by this repo, clips to the T800 MJCF limits, and writes GMR-compatible pickle
files.  Optional CSV output uses the same root quaternion convention as the
existing ``whole_body_tracking/scripts/t800_csv_to_npz.py`` converter:
``root_pos(xyz), root_quat(xyzw), t800_dof_pos``.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = Path("/data2/yangky/test/datasets/MotionDecode")
DEFAULT_OUTPUT_ROOT = Path("/data2/yangky/test/datasets/MotionDecode_T800")
DEFAULT_T800_XML = REPO_ROOT / "assets" / "t800" / "t800.xml"
DEFAULT_CATEGORIES = ("samples/4.Martial_Arts", "samples/5.Dance")

ROOT_WXYZ_COLUMNS = (
    "root_pos_x(m)",
    "root_pos_y(m)",
    "root_pos_z(m)",
    "root_rot_w",
    "root_rot_x",
    "root_rot_y",
    "root_rot_z",
)

# Target joint order matches whole_body_tracking.tasks.tracking.config.t800.t800_mdp.
# T800 has no waist roll/pitch or wrist joints in this 25-DoF model.  Wrist yaw
# is used as the closest available forearm-yaw signal; head joints are held at 0.
T800_JOINT_MAP = (
    ("J00_HIP_PITCH_L", "dof_left_hip_pitch_joint(rad)", 1.0),
    ("J01_HIP_ROLL_L", "dof_left_hip_roll_joint(rad)", 1.0),
    ("J02_HIP_YAW_L", "dof_left_hip_yaw_joint(rad)", 1.0),
    ("J03_KNEE_PITCH_L", "dof_left_knee_joint(rad)", 1.0),
    ("J04_ANKLE_PITCH_L", "dof_left_ankle_pitch_joint(rad)", 1.0),
    ("J05_ANKLE_ROLL_L", "dof_left_ankle_roll_joint(rad)", 1.0),
    ("J06_HIP_PITCH_R", "dof_right_hip_pitch_joint(rad)", 1.0),
    ("J07_HIP_ROLL_R", "dof_right_hip_roll_joint(rad)", 1.0),
    ("J08_HIP_YAW_R", "dof_right_hip_yaw_joint(rad)", 1.0),
    ("J09_KNEE_PITCH_R", "dof_right_knee_joint(rad)", 1.0),
    ("J10_ANKLE_PITCH_R", "dof_right_ankle_pitch_joint(rad)", 1.0),
    ("J11_ANKLE_ROLL_R", "dof_right_ankle_roll_joint(rad)", 1.0),
    ("J12_TORSO_YAW", "dof_waist_yaw_joint(rad)", 1.0),
    ("J13_SHOULDER_PITCH_L", "dof_left_shoulder_pitch_joint(rad)", 1.0),
    ("J14_SHOULDER_ROLL_L", "dof_left_shoulder_roll_joint(rad)", 1.0),
    ("J15_SHOULDER_YAW_L", "dof_left_shoulder_yaw_joint(rad)", 1.0),
    ("J16_ELBOW_PITCH_L", "dof_left_elbow_joint(rad)", -1.0),
    ("J17_ELBOW_YAW_L", "dof_left_wrist_yaw_joint(rad)", 1.0),
    ("J20_SHOULDER_PITCH_R", "dof_right_shoulder_pitch_joint(rad)", 1.0),
    ("J21_SHOULDER_ROLL_R", "dof_right_shoulder_roll_joint(rad)", 1.0),
    ("J22_SHOULDER_YAW_R", "dof_right_shoulder_yaw_joint(rad)", 1.0),
    ("J23_ELBOW_PITCH_R", "dof_right_elbow_joint(rad)", -1.0),
    ("J24_ELBOW_YAW_R", "dof_right_wrist_yaw_joint(rad)", 1.0),
    ("J27_HEAD_PITCH", None, 0.0),
    ("J28_HEAD_YAW", None, 0.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="Relative directories below input-root to convert.",
    )
    parser.add_argument("--t800-xml", type=Path, default=DEFAULT_T800_XML)
    parser.add_argument(
        "--fps",
        type=float,
        default=120.0,
        help="FPS to store in GMR pickle metadata. MotionDecode documents the release at 120 Hz.",
    )
    parser.add_argument("--write-csv", action="store_true", help="Also write 32-column T800 CSV files.")
    parser.add_argument("--csv-header", action="store_true", help="Include a CSV header row for human inspection.")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip joint angles to T800 limits.")
    parser.add_argument(
        "--align-floor",
        action="store_true",
        help="Apply one fixed root-z offset per motion so the lowest T800 collision geom touches the floor.",
    )
    parser.add_argument("--floor-align-model", type=Path, default=DEFAULT_T800_XML)
    parser.add_argument("--floor-height", type=float, default=0.0)
    parser.add_argument("--floor-clearance", type=float, default=0.005)
    parser.add_argument("--floor-align-stride", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--max-files", type=int, default=None, help="Convert only the first N files.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be converted without writing files.")
    return parser.parse_args()


def repo_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def load_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def load_motion_csv(csv_path: Path) -> tuple[list[str], np.ndarray]:
    header = load_header(csv_path)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(header):
        raise ValueError(f"{csv_path}: header has {len(header)} columns but data has {data.shape[1]}")
    return header, data


def parse_t800_limits(xml_path: Path) -> dict[str, tuple[float, float]]:
    tree = ET.parse(xml_path)
    limits: dict[str, tuple[float, float]] = {}
    for joint in tree.getroot().iter("joint"):
        name = joint.get("name")
        range_text = joint.get("range")
        if not name or not range_text:
            continue
        low, high = (float(part) for part in range_text.split())
        limits[name] = (low, high)
    return limits


def geom_bottom_z(model, data, geom_id: int) -> float:
    import mujoco as mj

    geom_type = int(model.geom_type[geom_id])
    pos = data.geom_xpos[geom_id]
    size = model.geom_size[geom_id]
    mat = data.geom_xmat[geom_id].reshape(3, 3)

    if geom_type == mj.mjtGeom.mjGEOM_SPHERE:
        return float(pos[2] - size[0])
    if geom_type == mj.mjtGeom.mjGEOM_BOX:
        return float(pos[2] - np.abs(mat[2, :]).dot(size[:3]))
    if geom_type in (mj.mjtGeom.mjGEOM_CAPSULE, mj.mjtGeom.mjGEOM_CYLINDER):
        radius = size[0]
        half_length = size[1]
        local_extent = np.array([radius, radius, half_length], dtype=np.float64)
        return float(pos[2] - np.abs(mat[2, :]).dot(local_extent))
    if geom_type == mj.mjtGeom.mjGEOM_PLANE:
        return float("inf")
    return float(pos[2] - float(np.max(size)))


def compute_floor_alignment(
    model,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    floor_height: float,
    clearance: float,
    stride: int,
) -> dict[str, float]:
    import mujoco as mj

    if model.nq != dof_pos.shape[1] + 7:
        raise ValueError(f"Floor align model expects {model.nq - 7} dofs, got {dof_pos.shape[1]}")
    if stride <= 0:
        raise ValueError("--floor-align-stride must be positive")

    data = mj.MjData(model)
    min_z = float("inf")
    min_frame = 0
    for frame_i in range(0, root_pos.shape[0], stride):
        data.qpos[:3] = root_pos[frame_i]
        data.qpos[3:7] = root_rot_xyzw[frame_i, [3, 0, 1, 2]]
        data.qpos[7:] = dof_pos[frame_i]
        mj.mj_forward(model, data)
        frame_min = min(geom_bottom_z(model, data, geom_id) for geom_id in range(model.ngeom))
        if frame_min < min_z:
            min_z = frame_min
            min_frame = frame_i

    target_min_z = floor_height + clearance
    offset_z = target_min_z - min_z
    root_pos[:, 2] += offset_z
    return {
        "floor_min_z_before": float(min_z),
        "floor_target_min_z": float(target_min_z),
        "floor_root_z_offset_m": float(offset_z),
        "floor_min_frame": int(min_frame),
    }


def output_path_for(output_root: Path, subdir: str, rel_csv: Path, suffix: str) -> Path:
    return output_root / subdir / rel_csv.with_name(f"{rel_csv.stem}_t800{suffix}")


def discover_csvs(input_root: Path, categories: list[str]) -> list[Path]:
    files: list[Path] = []
    for category in categories:
        category_path = input_root / category
        if not category_path.is_dir():
            raise FileNotFoundError(f"Category directory not found: {category_path}")
        files.extend(category_path.rglob("*.csv"))
    return sorted(files)


def convert_motion(
    csv_path: Path,
    input_root: Path,
    output_root: Path,
    limits: dict[str, tuple[float, float]],
    fps: float,
    write_csv: bool,
    csv_header: bool,
    clip: bool,
    floor_model,
    floor_height: float,
    floor_clearance: float,
    floor_align_stride: int,
    force: bool,
    dry_run: bool,
) -> dict:
    rel_csv = csv_path.relative_to(input_root)
    pkl_path = output_path_for(output_root, "gmr_pkl", rel_csv, ".pkl")
    out_csv_path = output_path_for(output_root, "csv_xyzw", rel_csv, ".csv") if write_csv else None

    if not force and pkl_path.exists() and (not write_csv or (out_csv_path and out_csv_path.exists())):
        return {
            "source": rel_csv.as_posix(),
            "pkl": pkl_path.relative_to(output_root).as_posix(),
            "csv": out_csv_path.relative_to(output_root).as_posix() if out_csv_path else None,
            "skipped": True,
        }

    header, data = load_motion_csv(csv_path)
    header_index = {name: i for i, name in enumerate(header)}
    missing_root = [name for name in ROOT_WXYZ_COLUMNS if name not in header_index]
    if missing_root:
        raise KeyError(f"{csv_path}: missing root columns: {missing_root}")

    root_pos = data[:, [header_index[name] for name in ROOT_WXYZ_COLUMNS[:3]]].astype(np.float32)
    root_rot_wxyz = data[:, [header_index[name] for name in ROOT_WXYZ_COLUMNS[3:]]].astype(np.float32)
    quat_norm = np.linalg.norm(root_rot_wxyz, axis=1, keepdims=True)
    quat_norm[quat_norm == 0.0] = 1.0
    root_rot_wxyz = root_rot_wxyz / quat_norm
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]

    target_joint_names = [entry[0] for entry in T800_JOINT_MAP]
    dof_pos = np.zeros((data.shape[0], len(T800_JOINT_MAP)), dtype=np.float32)
    for i, (_, source_column, sign) in enumerate(T800_JOINT_MAP):
        if source_column is None:
            continue
        if source_column not in header_index:
            raise KeyError(f"{csv_path}: missing source column {source_column}")
        dof_pos[:, i] = sign * data[:, header_index[source_column]]

    clip_summary: dict[str, dict[str, float | int]] = {}
    clip_count = 0
    if clip:
        lows = np.array([limits[name][0] for name in target_joint_names], dtype=np.float32)
        highs = np.array([limits[name][1] for name in target_joint_names], dtype=np.float32)
        clipped = np.clip(dof_pos, lows, highs)
        mask = np.abs(clipped - dof_pos) > 1e-6
        clip_count = int(mask.sum())
        if clip_count:
            over_low = np.maximum(lows - dof_pos, 0.0)
            over_high = np.maximum(dof_pos - highs, 0.0)
            over = np.maximum(over_low, over_high)
            for i, name in enumerate(target_joint_names):
                joint_count = int(mask[:, i].sum())
                if joint_count:
                    clip_summary[name] = {
                        "count": joint_count,
                        "max_violation_rad": float(over[:, i].max()),
                    }
        dof_pos = clipped.astype(np.float32)

    floor_alignment = None
    if floor_model is not None:
        floor_alignment = compute_floor_alignment(
            model=floor_model,
            root_pos=root_pos,
            root_rot_xyzw=root_rot_xyzw,
            dof_pos=dof_pos,
            floor_height=floor_height,
            clearance=floor_clearance,
            stride=floor_align_stride,
        )

    motion_data = {
        "fps": float(fps),
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw.astype(np.float32),
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
        "joint_names": target_joint_names,
        "source_file": str(csv_path),
        "source_format": "MotionDecode Unitree G1 CSV",
        "root_rot_format": "xyzw",
        "floor_alignment": floor_alignment,
    }

    if not dry_run:
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with pkl_path.open("wb") as f:
            pickle.dump(motion_data, f)

        if out_csv_path is not None:
            out_csv_path.parent.mkdir(parents=True, exist_ok=True)
            qpos = np.concatenate([root_pos, root_rot_xyzw, dof_pos], axis=1)
            header_text = None
            comments = "# "
            if csv_header:
                header_text = ",".join(
                    ["root_pos_x(m)", "root_pos_y(m)", "root_pos_z(m)", "root_rot_x", "root_rot_y", "root_rot_z", "root_rot_w"]
                    + [f"dof_{name}(rad)" for name in target_joint_names]
                )
                comments = ""
            np.savetxt(out_csv_path, qpos, delimiter=",", fmt="%.8f", header=header_text or "", comments=comments)

    duration = (data.shape[0] - 1) / fps if data.shape[0] > 1 else 0.0
    result = {
        "source": rel_csv.as_posix(),
        "pkl": pkl_path.relative_to(output_root).as_posix(),
        "csv": out_csv_path.relative_to(output_root).as_posix() if out_csv_path else None,
        "frames": int(data.shape[0]),
        "duration_sec": float(duration),
        "clip_count": clip_count,
        "clip_fraction": float(clip_count / max(1, data.shape[0] * len(T800_JOINT_MAP))),
        "clipped_joints": clip_summary,
        "skipped": False,
    }
    if floor_alignment is not None:
        result.update(floor_alignment)
    return result


def write_manifests(
    output_root: Path,
    input_root: Path,
    t800_xml: Path,
    fps: float,
    motions: list[dict],
    source_commit: str | None,
    write_csv: bool,
    align_floor: bool,
    floor_clearance: float,
    dry_run: bool,
) -> None:
    target_joint_names = [entry[0] for entry in T800_JOINT_MAP]
    tracking_input_fps = int(fps) if float(fps).is_integer() else float(fps)
    manifest = {
        "source_dataset": "https://huggingface.co/datasets/CMRobot/MotionDecode",
        "source_commit": source_commit,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "target_robot": "t800",
        "target_model": str(t800_xml),
        "fps": float(fps),
        "root_rot_format_in_pkl": "xyzw",
        "root_rot_format_in_csv": "xyzw" if write_csv else None,
        "floor_aligned": bool(align_floor),
        "floor_clearance_m": float(floor_clearance) if align_floor else None,
        "target_joint_names": target_joint_names,
        "source_to_target_joint_map": [
            {"target": target, "source": source, "sign": sign} for target, source, sign in T800_JOINT_MAP
        ],
        "notes": [
            "MotionDecode samples are Unitree G1 CSV trajectories.",
            "T800 25-DoF model has no waist roll/pitch or wrist joints; those G1 columns are not exported.",
            "G1 elbow flexion is sign-flipped to match T800 elbow pitch limits.",
            "Head pitch/yaw are held at zero because the source CSV has no head joints.",
            "When floor_aligned is true, each motion uses one fixed root-z offset so its lowest sampled T800 collision geometry is at floor_height + floor_clearance.",
        ],
        "motions": motions,
    }

    tracking_manifest = {
        "motions": [
            {
                "name": Path(motion["pkl"]).stem,
                "group": motion["source"].split("/")[1] if "/" in motion["source"] else "motiondecode",
                "input_file": str(output_root / motion["pkl"]),
                "input_format": "gmr_pickle",
                "input_fps": tracking_input_fps,
                "output_path": str(output_root / "tracking_npz" / Path(motion["source"]).with_name(f"{Path(motion['source']).stem}_t800_tracking.npz")),
            }
            for motion in motions
            if motion.get("pkl")
        ]
    }

    if dry_run:
        print(json.dumps({"manifest_preview": manifest, "tracking_manifest_preview": tracking_manifest}, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_root / "tracking_manifest.json").write_text(json.dumps(tracking_manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    t800_xml = args.t800_xml.expanduser().resolve()
    limits = parse_t800_limits(t800_xml)
    floor_model = None
    if args.align_floor:
        import mujoco as mj

        floor_model_path = args.floor_align_model.expanduser().resolve()
        floor_model = mj.MjModel.from_xml_path(str(floor_model_path))

    missing_limits = [name for name, _, _ in T800_JOINT_MAP if name not in limits]
    if missing_limits:
        raise KeyError(f"T800 XML is missing limits for target joints: {missing_limits}")

    csv_files = discover_csvs(input_root, args.categories)
    if args.max_files is not None:
        csv_files = csv_files[: args.max_files]

    motions: list[dict] = []
    total = len(csv_files)
    print(f"[INFO] Converting {total} MotionDecode CSV files from {input_root}")
    for i, csv_path in enumerate(csv_files, start=1):
        motion = convert_motion(
            csv_path=csv_path,
            input_root=input_root,
            output_root=output_root,
            limits=limits,
            fps=args.fps,
            write_csv=args.write_csv,
            csv_header=args.csv_header,
            clip=not args.no_clip,
            floor_model=floor_model,
            floor_height=args.floor_height,
            floor_clearance=args.floor_clearance,
            floor_align_stride=args.floor_align_stride,
            force=args.force,
            dry_run=args.dry_run,
        )
        motions.append(motion)
        if i == 1 or i == total or i % 50 == 0:
            status = "skip" if motion.get("skipped") else "ok"
            print(f"[{i:04d}/{total:04d}] {status} {motion['source']}")

    write_manifests(
        output_root=output_root,
        input_root=input_root,
        t800_xml=t800_xml,
        fps=args.fps,
        motions=motions,
        source_commit=repo_commit(input_root),
        write_csv=args.write_csv,
        align_floor=args.align_floor,
        floor_clearance=args.floor_clearance,
        dry_run=args.dry_run,
    )

    converted = sum(1 for motion in motions if not motion.get("skipped"))
    clipped = sum(int(motion.get("clip_count", 0)) for motion in motions)
    print(f"[OK] converted={converted} skipped={total - converted} total={total} clipped_values={clipped}")
    if not args.dry_run:
        print(f"[OK] wrote manifest: {output_root / 'manifest.json'}")
        print(f"[OK] wrote tracking manifest: {output_root / 'tracking_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
