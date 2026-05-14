#!/usr/bin/env python3
"""Convenience wrapper for T800 tracking playback."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Play a T800 tracking checkpoint with T800-friendly defaults.")
    parser.add_argument(
        "--load_run",
        type=str,
        required=True,
        help="Run folder name under logs/rsl_rl/t800_flat, or the full path to that run directory.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename, e.g. model_200.pt.")
    parser.add_argument("--motion_file", type=str, default=None, help="Optional motion npz override.")
    parser.add_argument("--num_envs", type=int, default=25, help="Number of playback environments.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for playback.")
    parser.add_argument(
        "--task_variant",
        choices=("default", "low_freq", "wo_state_estimation"),
        default="default",
        help="Which registered T800 task variant to launch.",
    )
    parser.add_argument("--video", action="store_true", help="Record a playback video.")
    parser.add_argument("--video_length", type=int, default=800, help="Playback video length in env steps.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without UI.")
    parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric USD acceleration.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded directly to scripts/rsl_rl/play.py",
    )
    args = parser.parse_args()

    task_map = {
        "default": "Tracking-Flat-T800-v0",
        "low_freq": "Tracking-Flat-T800-Low-Freq-v0",
        "wo_state_estimation": "Tracking-Flat-T800-Wo-State-Estimation-v0",
    }

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    play_script = repo_root / "scripts" / "rsl_rl" / "play.py"

    normalized_run = pathlib.Path(args.load_run).name

    cmd = [
        sys.executable,
        str(play_script),
        "--task",
        task_map[args.task_variant],
        "--load_run",
        normalized_run,
        "--num_envs",
        str(args.num_envs),
        "--device",
        args.device,
    ]

    if args.checkpoint is not None:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.motion_file is not None:
        cmd.extend(["--motion_file", args.motion_file])
    if args.video:
        cmd.extend(["--video", "--video_length", str(args.video_length)])
    if args.headless:
        cmd.append("--headless")
    if args.disable_fabric:
        cmd.append("--disable_fabric")

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)

    print("[INFO] Launching T800 playback command:")
    if normalized_run != args.load_run:
        print(f"[INFO] Normalized load_run to local run name: {normalized_run}")
    print(" ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
