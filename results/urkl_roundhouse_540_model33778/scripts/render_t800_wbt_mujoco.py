#!/usr/bin/env python3
"""Render an imported T800 WBT ONNX policy to an H.264 MuJoCo video."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("t800_wbt_sim2sim_imported", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path)
    parser.add_argument("--motion-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=0, help="Zero renders the complete motion.")
    args = parser.parse_args()

    runner = load_runner(args.runner.resolve())
    cfg = runner.load_config(
        args.config.resolve(),
        str(args.policy_dir.resolve()) if args.policy_dir else None,
        str(args.motion_file.resolve()) if args.motion_file else None,
    )
    runner.validate_config(cfg)

    session = runner._load_onnx_session(cfg.paths.policy_onnx)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    sim = runner.T800Sim2Sim(cfg)
    sim.reset()
    steps = args.steps or runner.motion_step_limit(cfg, sim)
    steps = min(steps, runner.motion_step_limit(cfg, sim))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args.width}x{args.height}",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output.resolve()),
    ]

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 90.0
    camera.elevation = -12.0
    camera.distance = 3.7
    sim.model.vis.global_.offwidth = args.width
    sim.model.vis.global_.offheight = args.height
    renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width)
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin was not created")

    try:
        for step in range(steps):
            runner._policy_step(session, input_name, output_name, sim, cfg, step)
            base_pos = sim.data.xpos[sim.base_body_id]
            camera.lookat[:] = np.array([base_pos[0], base_pos[1], 0.85], dtype=np.float64)
            renderer.update_scene(sim.data, camera=camera)
            process.stdin.write(renderer.render().tobytes())
    finally:
        renderer.close()
        process.stdin.close()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    print(
        f"rendered={args.output.resolve()} frames={steps} "
        f"duration_s={steps / args.fps:.2f} mujoco_gl={os.environ.get('MUJOCO_GL', 'default')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
