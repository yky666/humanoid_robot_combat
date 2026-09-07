#!/usr/bin/env python3
"""Render T800 SDK joint logs in MuJoCo for visual inspection."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco as mj
import numpy as np


WBT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WBT_ROOT.parent
sys.path.insert(0, str(WBT_ROOT / "source" / "whole_body_tracking"))

from whole_body_tracking.robots.t800_joint_order import (  # noqa: E402
    T800_POLICY_JOINT_NAMES,
    T800_SDK_POLICY_JOINT_NAMES,
)


DEFAULT_ARCHIVE = Path("/mnt/data/yangky/test/datasets/urkl_locomotion_260901/motion_logs.zip")
DEFAULT_MOTION = "logs/pdstand2baoquan.csv"
DEFAULT_XML = REPO_ROOT / "GMR" / "assets" / "t800" / "t800_visual.xml"
VIEW_AZIMUTHS = {
    "front": 180.0,
    "right": 90.0,
    "left": -90.0,
    "back": 0.0,
    "quarter": 135.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ARCHIVE, help="CSV file or zip archive.")
    parser.add_argument("--member", default=DEFAULT_MOTION, help="CSV member when --input is a zip archive.")
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--views", default="front,right", help="Comma-separated views: front,right,left,back,quarter.")
    parser.add_argument("--root-height", type=float, default=0.86)
    parser.add_argument("--floor-align", choices=("none", "each"), default="each")
    parser.add_argument("--camera-distance", type=float, default=2.5)
    parser.add_argument("--camera-elevation", type=float, default=-12.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 means full sequence.")
    parser.add_argument("--tail-hold-seconds", type=float, default=1.0)
    return parser.parse_args()


def read_csv_text(path: Path, member: str) -> str:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as handle:
                return handle.read().decode("utf-8-sig")
    return path.read_text(encoding="utf-8-sig")


def load_sdk_positions(path: Path, member: str) -> tuple[np.ndarray, np.ndarray]:
    reader = csv.reader(io.StringIO(read_csv_text(path, member)))
    header = next(reader)
    rows = [[float(value) for value in row] for row in reader if row]
    if not rows:
        raise ValueError(f"{path}:{member} contains no samples")
    data = np.asarray(rows, dtype=np.float64)
    columns = {name: index for index, name in enumerate(header)}
    missing = [f"pos_{name}" for name in T800_SDK_POLICY_JOINT_NAMES if f"pos_{name}" not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    time_s = data[:, columns["t_host"]]
    joint_pos_policy = data[:, [columns[f"pos_{name}"] for name in T800_SDK_POLICY_JOINT_NAMES]]
    return time_s, joint_pos_policy


def sampling_rate(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def make_scene_xml(xml_path: Path) -> tuple[Path, Path | None]:
    text = xml_path.read_text(encoding="utf-8")
    if "<worldbody>" not in text:
        return xml_path, None
    scene_items = (
        '<light name="render_key" pos="0 -3 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>\n'
        '    <light name="render_fill" pos="-3 2 3" dir="0 0 -1" diffuse="0.35 0.35 0.35"/>\n'
        '    <geom name="render_floor" type="plane" size="6 6 0.01" rgba="0.86 0.86 0.82 1"/>'
    )
    text = text.replace("<worldbody>", f"<worldbody>\n    {scene_items}", 1)
    handle = tempfile.NamedTemporaryFile("w", suffix=".xml", prefix=".t800_real_log_", dir=xml_path.parent, delete=False)
    try:
        handle.write(text)
    finally:
        handle.close()
    return Path(handle.name), Path(handle.name)


def parse_views(views_text: str) -> list[tuple[str, float]]:
    output = []
    for raw in views_text.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in VIEW_AZIMUTHS:
            raise ValueError(f"Unknown view {name!r}; valid views: {sorted(VIEW_AZIMUTHS)}")
        output.append((name, VIEW_AZIMUTHS[name]))
    if not output:
        raise ValueError("At least one view is required")
    return output


def frame_indices(frame_count: int, source_fps: float, render_fps: float, start_seconds: float, max_seconds: float) -> np.ndarray:
    start = max(0, min(frame_count - 1, int(round(start_seconds * source_fps)))) if source_fps > 0 else 0
    if max_seconds > 0 and source_fps > 0:
        end = min(frame_count, start + max(1, int(round(max_seconds * source_fps))))
    else:
        end = frame_count
    step = max(1, int(round(source_fps / render_fps))) if source_fps > 0 else 1
    indices = np.arange(start, end, step, dtype=np.int64)
    if len(indices) == 0:
        indices = np.asarray([start], dtype=np.int64)
    return indices


def start_ffmpeg(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def lowest_geom_z(model: mj.MjModel, data: mj.MjData) -> float:
    lowest = float("inf")
    for geom_id in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or ""
        geom_type = int(model.geom_type[geom_id])
        if name == "render_floor" or geom_type == mj.mjtGeom.mjGEOM_PLANE:
            continue
        center = np.asarray(data.geom_xpos[geom_id])
        if geom_type == mj.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id >= 0:
                start = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                vertices = model.mesh_vert[start : start + count]
                xmat = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
                world_z = center[2] + vertices @ xmat[2, :]
                lowest = min(lowest, float(np.min(world_z)))
            continue
        size = np.asarray(model.geom_size[geom_id])
        if geom_type == mj.mjtGeom.mjGEOM_BOX:
            xmat = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
            extent_z = float(np.sum(np.abs(xmat[2, :]) * size[:3]))
            lowest = min(lowest, center[2] - extent_z)
        else:
            lowest = min(lowest, center[2] - float(size[0]))
    return 0.0 if not np.isfinite(lowest) else lowest


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    xml_path = args.robot_xml.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    metadata_path = args.metadata_json.expanduser().resolve() if args.metadata_json else None

    time_s, joint_pos = load_sdk_positions(input_path, args.member)
    source_fps = sampling_rate(time_s)
    indices = frame_indices(len(time_s), source_fps, args.render_fps, args.start_seconds, args.max_seconds)
    video_fps = source_fps / max(1, int(round(source_fps / args.render_fps))) if source_fps > 0 else args.render_fps

    hold_frames = max(0, int(round(args.tail_hold_seconds * video_fps)))
    if hold_frames:
        indices = np.concatenate([indices, np.full(hold_frames, indices[-1], dtype=np.int64)])

    scene_xml, temporary_xml = make_scene_xml(xml_path)
    try:
        model = mj.MjModel.from_xml_path(str(scene_xml))
    finally:
        if temporary_xml is not None:
            temporary_xml.unlink(missing_ok=True)

    expected = len(T800_POLICY_JOINT_NAMES)
    if model.nq - 7 != expected:
        raise ValueError(f"{xml_path}: expected {expected} joints plus free root, got nq={model.nq}")
    joint_qpos_addr = []
    for joint_name in T800_POLICY_JOINT_NAMES:
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"{xml_path}: missing MuJoCo joint {joint_name}")
        joint_qpos_addr.append(int(model.jnt_qposadr[joint_id]))

    data = mj.MjData(model)
    renderer = mj.Renderer(model, width=args.width, height=args.height)
    views = parse_views(args.views)
    cameras = []
    for _, azimuth in views:
        camera = mj.MjvCamera()
        mj.mjv_defaultCamera(camera)
        camera.azimuth = azimuth
        camera.elevation = args.camera_elevation
        camera.distance = args.camera_distance
        camera.lookat[:] = np.asarray([0.0, 0.0, 0.55])
        cameras.append(camera)

    panel_width = args.width * len(views)
    proc = start_ffmpeg(output_path, panel_width, args.height, video_fps)
    try:
        if proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        for frame_i in indices:
            data.qpos[:] = 0.0
            data.qpos[0:3] = np.asarray([0.0, 0.0, args.root_height])
            data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
            for value, addr in zip(joint_pos[frame_i], joint_qpos_addr, strict=True):
                data.qpos[addr] = value
            mj.mj_forward(model, data)
            if args.floor_align == "each":
                data.qpos[2] -= lowest_geom_z(model, data)
                mj.mj_forward(model, data)

            panels = []
            for camera in cameras:
                renderer.update_scene(data, camera=camera)
                panels.append(renderer.render())
            frame = np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0]
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        return_code = proc.wait()
        renderer.close()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    metadata = {
        "source": str(input_path),
        "member": args.member,
        "robot_xml": str(xml_path),
        "output": str(output_path),
        "source_frames": int(len(time_s)),
        "source_duration_s": float(time_s[-1] - time_s[0]),
        "source_estimated_fps": source_fps,
        "rendered_frames": int(len(indices)),
        "render_fps": float(video_fps),
        "tail_hold_seconds": float(args.tail_hold_seconds),
        "views": [name for name, _ in views],
        "root_height": float(args.root_height),
        "floor_align": args.floor_align,
        "joint_names": T800_POLICY_JOINT_NAMES,
        "source_joint_names": T800_SDK_POLICY_JOINT_NAMES,
    }
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] wrote {metadata_path}")
    print(f"[OK] wrote {output_path}")
    print(f"[INFO] rendered_frames={metadata['rendered_frames']} render_fps={video_fps:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
