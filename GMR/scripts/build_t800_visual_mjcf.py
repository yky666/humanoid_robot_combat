#!/usr/bin/env python3
"""Build a visual T800 MJCF for MuJoCo rendering.

The bundled T800 URDF references Collada ``.dae`` visual meshes, while this
environment's MuJoCo build cannot decode those DAE files directly.  This helper
uses Blender to convert the meshes to OBJ, imports the URDF with visual geoms
enabled, saves MJCF, then adds a root freejoint so retargeted root trajectories
can be replayed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import mujoco as mj


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "assets" / "t800" / "serial_t800.urdf"
DEFAULT_OBJ_DIR = REPO_ROOT / "assets" / "t800" / "meshes_obj"
DEFAULT_OUTPUT_XML = REPO_ROOT / "assets" / "t800" / "t800_visual.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--obj-dir", type=Path, default=DEFAULT_OBJ_DIR)
    parser.add_argument("--output-xml", type=Path, default=DEFAULT_OUTPUT_XML)
    parser.add_argument("--blender", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-mesh-convert", action="store_true")
    return parser.parse_args()


def blender_path(requested: Path | None) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    found = shutil.which("blender")
    if found:
        return Path(found)
    fallback = Path("/usr/local/bin/blender")
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Blender not found. Pass --blender /path/to/blender.")


def mesh_tasks(urdf: Path, obj_dir: Path, force: bool) -> list[tuple[Path, Path]]:
    dae_files = sorted((urdf.parent / "meshes").glob("*.dae"))
    if not dae_files:
        raise FileNotFoundError(f"No DAE files found below {urdf.parent / 'meshes'}")
    tasks = []
    for dae in dae_files:
        obj = obj_dir / f"{dae.stem}.obj"
        if force or not obj.exists() or obj.stat().st_mtime < dae.stat().st_mtime:
            tasks.append((dae, obj))
    return tasks


def convert_meshes(blender: Path, tasks: list[tuple[Path, Path]]) -> None:
    if not tasks:
        print("[INFO] OBJ mesh cache is up to date")
        return
    for _, dst in tasks:
        dst.parent.mkdir(parents=True, exist_ok=True)

    blender_script = r"""
import json
import sys

import bpy

payload = json.loads(sys.argv[sys.argv.index("--") + 1])

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def export_obj(filepath):
    bpy.ops.object.select_all(action="SELECT")
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=filepath,
            export_selected_objects=True,
            export_materials=False,
            forward_axis="Y",
            up_axis="Z",
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=filepath,
            use_selection=True,
            use_materials=False,
            axis_forward="Y",
            axis_up="Z",
        )

for src, dst in payload:
    clear_scene()
    bpy.ops.wm.collada_import(filepath=src)
    export_obj(dst)
    print(f"[BLENDER] {src} -> {dst}")
"""
    payload = json.dumps([(str(src), str(dst)) for src, dst in tasks])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(blender_script)
        script_path = Path(f.name)
    try:
        cmd = [str(blender), "-b", "--python", str(script_path), "--", payload]
        subprocess.run(cmd, check=True)
    finally:
        script_path.unlink(missing_ok=True)


def make_visual_urdf(urdf: Path, obj_dir: Path) -> Path:
    text = urdf.read_text(encoding="utf-8")
    text = re.sub(r'filename="(?:\.\./)?meshes/([^"]+)\.dae"', r'filename="\1.obj"', text)
    compiler = f'  <mujoco>\n    <compiler meshdir="{obj_dir}" discardvisual="false"/>\n  </mujoco>\n'
    if "<mujoco>" in text:
        text = re.sub(r"\s*<mujoco>.*?</mujoco>\s*", "\n" + compiler, text, flags=re.DOTALL, count=1)
    else:
        text = text.replace('<robot name="t800">', '<robot name="t800">\n' + compiler, 1)
    tmp = urdf.parent / ".t800_visual_import.urdf"
    tmp.write_text(text, encoding="utf-8")
    return tmp


def add_root_freejoint(raw_xml: Path, output_xml: Path) -> None:
    tree = ET.parse(raw_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{raw_xml}: missing worldbody")
    base = worldbody.find(".//body[@name='LINK_BASE']")
    if base is None:
        # MuJoCo's URDF importer keeps the root link geoms directly under
        # worldbody.  Wrap the imported tree into an explicit free-root body.
        base = ET.Element("body", {"name": "LINK_BASE"})
        children = list(worldbody)
        for child in children:
            worldbody.remove(child)
            base.append(child)
        worldbody.append(base)
    has_freejoint = any(child.tag == "freejoint" and child.get("name") == "root" for child in list(base))
    if not has_freejoint:
        base.insert(0, ET.Element("freejoint", {"name": "root"}))
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)


def validate_model(output_xml: Path) -> None:
    model = mj.MjModel.from_xml_path(str(output_xml))
    joints = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    mesh_geoms = sum(int(model.geom_type[i]) == mj.mjtGeom.mjGEOM_MESH for i in range(model.ngeom))
    print(f"[OK] wrote {output_xml}")
    print(f"[OK] nq={model.nq} nv={model.nv} njnt={model.njnt} ngeom={model.ngeom} nmesh={model.nmesh} mesh_geoms={mesh_geoms}")
    print(f"[OK] joints={joints[:3]} ... {joints[-3:]}")
    if model.nq != 32 or joints[0] != "root" or len(joints) != 26 or mesh_geoms == 0:
        raise RuntimeError("Generated visual MJCF does not match expected free-root T800 model.")


def main() -> int:
    args = parse_args()
    urdf = args.urdf.expanduser().resolve()
    obj_dir = args.obj_dir.expanduser().resolve()
    output_xml = args.output_xml.expanduser().resolve()

    if not args.skip_mesh_convert:
        tasks = mesh_tasks(urdf, obj_dir, args.force)
        print(f"[INFO] converting {len(tasks)} mesh file(s)")
        convert_meshes(blender_path(args.blender), tasks)

    visual_urdf = make_visual_urdf(urdf, obj_dir)
    raw_xml = output_xml.with_suffix(".raw.xml")
    try:
        model = mj.MjModel.from_xml_path(str(visual_urdf))
        mj.mj_saveLastXML(str(raw_xml), model)
        add_root_freejoint(raw_xml, output_xml)
    finally:
        visual_urdf.unlink(missing_ok=True)
        raw_xml.unlink(missing_ok=True)

    validate_model(output_xml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
