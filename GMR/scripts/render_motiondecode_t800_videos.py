#!/usr/bin/env python3
"""Render T800 retargeted motions to inspection videos.

The converter in ``motiondecode_g1_csv_to_t800.py`` writes GMR-style pickle
files and optional 32-column CSV files.  The IsaacLab/BeyondMimic preparation
step writes tracking ``.npz`` files.  This script renders those trajectories
with the T800 MuJoCo asset in an offscreen context so batches can be reviewed
without opening the interactive MuJoCo viewer.
"""

from __future__ import annotations

import argparse
import fnmatch
import html
import json
import os
import pickle
import tempfile
from pathlib import Path

# Set before importing mujoco.  Users can override with MUJOCO_GL=osmesa/glfw.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco as mj
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - overlay is optional.
    cv2 = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/data2/yangky/test/datasets/MotionDecode_T800")
DEFAULT_INPUT_ROOT = DEFAULT_DATA_ROOT / "gmr_pkl"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "videos"
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "manifest.json"
DEFAULT_T800_XML = REPO_ROOT / "assets" / "t800" / "t800.xml"
DEFAULT_T800_VISUAL_XML = REPO_ROOT / "assets" / "t800" / "t800_visual.xml"
T800_BASE_BODY = "LINK_BASE"
T800_CAMERA_DISTANCE = 2.5

VIEW_AZIMUTHS = {
    "front": 180.0,
    "back": 0.0,
    "left": -90.0,
    "right": 90.0,
    "quarter": 135.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--input-file", type=Path, action="append", default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=None,
        help="MJCF/URDF to render. Defaults to assets/t800/t800_visual.xml when available.",
    )
    parser.add_argument("--input-format", choices=("auto", "pkl", "csv", "tracking_npz", "gmr_npz"), default="auto")
    parser.add_argument("--source-fps", type=float, default=120.0, help="Fallback FPS for CSV input.")
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=None, help="Frame stride; overrides render-fps.")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 means render the full motion.")
    parser.add_argument("--width", type=int, default=640, help="Width per view.")
    parser.add_argument("--height", type=int, default=480, help="Height per view.")
    parser.add_argument("--views", type=str, default="front,right", help="Comma list: front,right,left,back,quarter.")
    parser.add_argument("--camera-distance", type=float, default=T800_CAMERA_DISTANCE)
    parser.add_argument("--camera-elevation", type=float, default=-12.0)
    parser.add_argument("--no-follow", action="store_true", help="Use one fixed look-at point per motion.")
    parser.add_argument("--no-floor", action="store_true", help="Do not inject a visual floor into the XML.")
    parser.add_argument("--no-overlay", action="store_true", help="Do not draw motion name/time overlay.")
    parser.add_argument("--show-collision-geoms", action="store_true", help="Keep non-mesh collision geoms visible.")
    parser.add_argument("--category", action="append", default=None, help="Keep paths containing this substring.")
    parser.add_argument("--pattern", action="append", default=None, help="fnmatch pattern on the relative path.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--index-file", type=Path, default=None)
    return parser.parse_args()


def default_robot_xml() -> Path:
    if DEFAULT_T800_VISUAL_XML.is_file():
        return DEFAULT_T800_VISUAL_XML
    return DEFAULT_T800_XML


def load_manifest_files(manifest_path: Path, input_root: Path) -> tuple[list[Path], Path]:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    data_root = manifest_path.parent
    files = []
    for motion in manifest.get("motions", []):
        rel_pkl = motion.get("pkl")
        if rel_pkl:
            files.append(data_root / rel_pkl)
    base_root = input_root
    return files, base_root


def discover_inputs(args: argparse.Namespace) -> tuple[list[Path], Path]:
    input_root = args.input_root.expanduser().resolve()
    if args.input_file:
        files = [path.expanduser().resolve() for path in args.input_file]
        return files, input_root

    manifest = args.manifest.expanduser().resolve()
    if manifest.is_file() and input_root == DEFAULT_INPUT_ROOT.resolve():
        files, base_root = load_manifest_files(manifest, input_root)
    else:
        if args.input_format == "auto":
            suffixes = (".pkl", ".csv", ".npz")
        elif args.input_format in {"tracking_npz", "gmr_npz"}:
            suffixes = (".npz",)
        else:
            suffixes = (f".{args.input_format}",)
        files = sorted(path for path in input_root.rglob("*") if path.suffix.lower() in suffixes)
        base_root = input_root

    filtered = []
    for path in files:
        try:
            rel = path.resolve().relative_to(base_root)
        except ValueError:
            rel = Path(path.name)
        rel_text = rel.as_posix()
        if args.category and not any(category in rel_text for category in args.category):
            continue
        if args.pattern and not any(fnmatch.fnmatch(rel_text, pattern) for pattern in args.pattern):
            continue
        filtered.append(path)

    if args.limit is not None:
        filtered = filtered[: args.limit]
    return filtered, base_root


def infer_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return "pkl"
    if suffix == ".csv":
        return "csv"
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
        if {"body_pos_w", "body_quat_w", "joint_pos"}.issubset(keys):
            return "tracking_npz"
        if {"root_pos", "root_rot", "dof_pos"}.issubset(keys):
            return "gmr_npz"
    raise ValueError(f"Cannot infer input format for {path}")


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return quat / norm


def load_motion(path: Path, input_format: str, fallback_fps: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    input_format = infer_format(path, input_format)
    if input_format == "pkl":
        with path.open("rb") as f:
            motion = pickle.load(f)
        fps = float(motion.get("fps", fallback_fps))
        root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
        root_rot = np.asarray(motion["root_rot"], dtype=np.float32)
        root_rot_format = str(motion.get("root_rot_format", "xyzw")).lower()
        if root_rot_format == "xyzw":
            root_rot_wxyz = root_rot[:, [3, 0, 1, 2]]
        elif root_rot_format == "wxyz":
            root_rot_wxyz = root_rot
        else:
            raise ValueError(f"{path}: unsupported root_rot_format={root_rot_format}")
        dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
    elif input_format == "csv":
        data = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] != 32:
            raise ValueError(f"{path}: expected 32 CSV columns, got {data.shape[1]}")
        fps = float(fallback_fps)
        root_pos = data[:, :3]
        root_rot_wxyz = data[:, [6, 3, 4, 5]]
        dof_pos = data[:, 7:]
    else:
        with np.load(path, allow_pickle=False) as motion:
            if input_format == "tracking_npz":
                fps = float(np.asarray(motion["fps"]).reshape(-1)[0]) if "fps" in motion.files else float(fallback_fps)
                body_pos = np.asarray(motion["body_pos_w"], dtype=np.float32)
                body_quat = np.asarray(motion["body_quat_w"], dtype=np.float32)
                if body_pos.ndim != 3 or body_pos.shape[1] < 1 or body_pos.shape[2] != 3:
                    raise ValueError(f"{path}: expected body_pos_w shape [frames, bodies, 3], got {body_pos.shape}")
                if body_quat.ndim != 3 or body_quat.shape[1] < 1 or body_quat.shape[2] != 4:
                    raise ValueError(f"{path}: expected body_quat_w shape [frames, bodies, 4], got {body_quat.shape}")
                root_pos = body_pos[:, 0, :]
                root_rot_wxyz = body_quat[:, 0, :]
                dof_pos = np.asarray(motion["joint_pos"], dtype=np.float32)
            elif input_format == "gmr_npz":
                fps = float(np.asarray(motion["fps"]).reshape(-1)[0]) if "fps" in motion.files else float(fallback_fps)
                root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
                root_rot = np.asarray(motion["root_rot"], dtype=np.float32)
                if "root_rot_format" in motion.files:
                    root_rot_format = str(np.asarray(motion["root_rot_format"]).reshape(-1)[0]).lower()
                else:
                    root_rot_format = "xyzw"
                if root_rot_format == "xyzw":
                    root_rot_wxyz = root_rot[:, [3, 0, 1, 2]]
                elif root_rot_format == "wxyz":
                    root_rot_wxyz = root_rot
                else:
                    raise ValueError(f"{path}: unsupported root_rot_format={root_rot_format}")
                dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
            else:
                raise ValueError(f"{path}: unsupported input_format={input_format}")

    if root_pos.shape[0] != root_rot_wxyz.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
        raise ValueError(f"{path}: root/quaternion/dof frame counts do not match")
    return fps, root_pos, normalize_quat(root_rot_wxyz), dof_pos


def output_path_for(path: Path, base_root: Path, output_root: Path) -> Path:
    try:
        rel = path.resolve().relative_to(base_root)
    except ValueError:
        rel = Path(path.name)
    return output_root / rel.with_suffix(".mp4")


def make_scene_xml(xml_path: Path, add_floor: bool) -> tuple[Path, Path | None]:
    if not add_floor:
        return xml_path, None
    text = xml_path.read_text(encoding="utf-8")
    scene_items = (
        '<light name="render_key" pos="0 -2 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>\n'
        '    <light name="render_fill" pos="-2 2 2" dir="0 0 -1" diffuse="0.35 0.35 0.35"/>\n'
        '    <geom name="render_floor" type="plane" size="6 6 0.01" rgba="0.88 0.88 0.84 1"/>'
    )
    if "<worldbody>" not in text:
        return xml_path, None
    text = text.replace("<worldbody>", f"<worldbody>\n    {scene_items}", 1)
    fd, tmp_name = tempfile.mkstemp(prefix=".render_t800_", suffix=".xml", dir=str(xml_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.write_text(text, encoding="utf-8")
    return tmp_path, tmp_path


def hide_nonmesh_geoms(model: mj.MjModel) -> None:
    if model.nmesh <= 0:
        return
    for geom_id in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "render_floor":
            continue
        if int(model.geom_type[geom_id]) != mj.mjtGeom.mjGEOM_MESH:
            model.geom_rgba[geom_id, 3] = 0.0


def select_frame_indices(
    frame_count: int,
    source_fps: float,
    render_fps: float,
    stride: int | None,
    start_seconds: float,
    max_seconds: float,
) -> tuple[np.ndarray, float]:
    if frame_count <= 0:
        raise ValueError("Motion has no frames")
    start = max(0, min(frame_count - 1, int(round(start_seconds * source_fps))))
    if max_seconds > 0:
        end = min(frame_count, start + max(1, int(round(max_seconds * source_fps))))
    else:
        end = frame_count
    if end <= start:
        end = min(frame_count, start + 1)
    step = stride if stride is not None else max(1, int(round(source_fps / render_fps)))
    if step <= 0:
        raise ValueError("--stride must be positive")
    indices = np.arange(start, end, step, dtype=np.int64)
    if len(indices) == 0:
        indices = np.array([start], dtype=np.int64)
    actual_fps = source_fps / step
    return indices, actual_fps


def parse_views(view_text: str) -> list[tuple[str, float]]:
    views = []
    for raw_name in view_text.split(","):
        name = raw_name.strip().lower()
        if not name:
            continue
        if name not in VIEW_AZIMUTHS:
            valid = ", ".join(sorted(VIEW_AZIMUTHS))
            raise ValueError(f"Unknown view '{name}'. Valid views: {valid}")
        views.append((name, VIEW_AZIMUTHS[name]))
    if not views:
        raise ValueError("At least one view is required")
    return views


def draw_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    if cv2 is None or not lines:
        return frame
    out = frame.copy()
    y = 24
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        y += 22
    return out


def render_motion(
    model: mj.MjModel,
    path: Path,
    out_path: Path,
    rel_text: str,
    args: argparse.Namespace,
    views: list[tuple[str, float]],
) -> dict:
    fps, root_pos, root_rot_wxyz, dof_pos = load_motion(path, args.input_format, args.source_fps)
    expected_dofs = model.nq - 7
    if dof_pos.shape[1] != expected_dofs:
        raise ValueError(f"{path}: expected {expected_dofs} dofs for model, got {dof_pos.shape[1]}")

    indices, video_fps = select_frame_indices(
        frame_count=len(root_pos),
        source_fps=fps,
        render_fps=args.render_fps,
        stride=args.stride,
        start_seconds=args.start_seconds,
        max_seconds=args.max_seconds,
    )

    if args.dry_run:
        print(f"[DRY] {rel_text} -> {out_path} frames={len(indices)} video_fps={video_fps:.2f}")
        return {"input": str(path), "output": str(out_path), "frames": int(len(indices)), "fps": float(video_fps)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = mj.MjData(model)
    renderer = mj.Renderer(model, width=args.width, height=args.height)
    cameras = []
    fixed_lookat = np.median(root_pos, axis=0)
    fixed_lookat[2] = max(0.5, fixed_lookat[2])
    base_body_id = model.body(T800_BASE_BODY).id
    for _, azimuth in views:
        cam = mj.MjvCamera()
        mj.mjv_defaultCamera(cam)
        cam.distance = args.camera_distance
        cam.elevation = args.camera_elevation
        cam.azimuth = azimuth
        cameras.append(cam)

    writer = imageio.get_writer(str(out_path), fps=video_fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        total = len(indices)
        for output_i, frame_i in enumerate(indices):
            data.qpos[:3] = root_pos[frame_i]
            data.qpos[3:7] = root_rot_wxyz[frame_i]
            data.qpos[7:] = dof_pos[frame_i]
            mj.mj_forward(model, data)

            panels = []
            for view_name, _ in views:
                cam = cameras[len(panels)]
                if args.no_follow:
                    cam.lookat[:] = fixed_lookat
                else:
                    cam.lookat[:] = data.xpos[base_body_id]
                renderer.update_scene(data, camera=cam)
                panel = renderer.render()
                if not args.no_overlay:
                    seconds = frame_i / fps
                    panel = draw_overlay(panel, [path.stem, f"{view_name}  t={seconds:.2f}s"])
                panels.append(panel)
            frame = np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0]
            writer.append_data(frame)

            if total >= 120 and (output_i + 1) % 120 == 0:
                print(f"    rendered {output_i + 1}/{total} frames for {rel_text}")
    finally:
        writer.close()
        renderer.close()

    return {
        "input": str(path),
        "output": str(out_path),
        "frames": int(len(indices)),
        "fps": float(video_fps),
        "bytes": out_path.stat().st_size,
    }


def write_index(index_file: Path, rendered: list[dict], output_root: Path) -> None:
    index_file.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for item in rendered:
        out_path = Path(item["output"])
        try:
            rel = out_path.relative_to(index_file.parent)
        except ValueError:
            rel = out_path
        title = html.escape(out_path.stem)
        src = html.escape(rel.as_posix())
        items.append(
            "<section>"
            f"<h2>{title}</h2>"
            f"<video controls preload=\"metadata\" src=\"{src}\"></video>"
            "</section>"
        )
    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>T800 MotionDecode Review</title>
<style>
body { margin: 24px; font-family: sans-serif; background: #f6f6f3; color: #181818; }
h1 { font-size: 24px; margin: 0 0 18px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }
section { min-width: 0; }
h2 { font-size: 14px; font-weight: 600; margin: 0 0 8px; overflow-wrap: anywhere; }
video { width: 100%; background: #111; border: 1px solid #ccc; }
</style>
</head>
<body>
<h1>T800 MotionDecode Review</h1>
<div class="grid">
""" + "\n".join(items) + """
</div>
</body>
</html>
"""
    index_file.write_text(page, encoding="utf-8")
    print(f"[OK] wrote index: {index_file}")


def main() -> int:
    args = parse_args()
    files, base_root = discover_inputs(args)
    if not files:
        print("[WARN] no matching motion files")
        return 0

    views = parse_views(args.views)
    output_root = args.output_root.expanduser().resolve()
    robot_xml = args.robot_xml if args.robot_xml is not None else default_robot_xml()
    xml_path, tmp_xml = make_scene_xml(robot_xml.expanduser().resolve(), add_floor=not args.no_floor)
    try:
        model = mj.MjModel.from_xml_path(str(xml_path))
    finally:
        if tmp_xml is not None:
            tmp_xml.unlink(missing_ok=True)
    if not args.show_collision_geoms:
        hide_nonmesh_geoms(model)

    rendered = []
    total = len(files)
    print(f"[INFO] rendering {total} motion(s), robot_xml={robot_xml}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    for i, path in enumerate(files, start=1):
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        out_path = output_path_for(path, base_root, output_root)
        try:
            rel_text = path.relative_to(base_root).as_posix()
        except ValueError:
            rel_text = path.name
        if out_path.exists() and not args.overwrite:
            print(f"[{i:04d}/{total:04d}] skip existing {out_path}")
            rendered.append({"input": str(path), "output": str(out_path), "skipped": True})
            continue
        print(f"[{i:04d}/{total:04d}] render {rel_text} -> {out_path}")
        rendered.append(render_motion(model, path, out_path, rel_text, args, views))

    index_file = args.index_file
    if index_file is None:
        index_file = output_root / "index.html"
    if not args.dry_run:
        write_index(index_file.expanduser().resolve(), [item for item in rendered if Path(item["output"]).is_file()], output_root)
    print(f"[OK] done: {len(rendered)} motion(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
