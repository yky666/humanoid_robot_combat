#!/usr/bin/env python3
"""Convenience wrapper for T800 tracking training.

This keeps the existing generic train.py untouched while providing
T800-friendly defaults for local motion training.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a T800 policy with T800-friendly defaults.")
    parser.add_argument("--motion_file", type=str, default=None, help="Path to a tracking-ready T800 motion npz.")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel environments.")
    parser.add_argument("--max_iterations", type=int, default=200, help="Number of PPO iterations.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for simulation/training.")
    parser.add_argument("--run_name", type=str, default="t800_probe", help="Run name shown in logs/W&B.")
    parser.add_argument("--log_project_name", type=str, default="t800_boxing", help="W&B project name.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--logger", type=str, default="wandb", help="Logger backend, e.g. wandb or tensorboard.")
    parser.add_argument("--getup_target_json", type=str, default=None, help="Measured target_joint_pos JSON for direct get-up tasks.")
    parser.add_argument(
        "--task_variant",
        choices=(
            "default",
            "recovery",
            "getup_prone",
            "getup_supine",
            "getup_mixed",
            "low_freq",
            "wo_state_estimation",
        ),
        default="default",
        help="Which registered T800 task variant to launch.",
    )
    parser.add_argument("--video", action="store_true", help="Record training videos.")
    parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in env steps.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between recorded videos.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without UI.")
    parser.add_argument("--enable_cameras", action="store_true", help="Enable cameras explicitly.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded directly to scripts/rsl_rl/train.py",
    )
    args = parser.parse_args()

    task_map = {
        "default": "Tracking-Flat-T800-v0",
        "recovery": "Tracking-Flat-T800-Recovery-v0",
        "getup_prone": "Getup-Direct-T800-Prone-v0",
        "getup_supine": "Getup-Direct-T800-Supine-v0",
        "getup_mixed": "Getup-Direct-T800-Mixed-v0",
        "low_freq": "Tracking-Flat-T800-Low-Freq-v0",
        "wo_state_estimation": "Tracking-Flat-T800-Wo-State-Estimation-v0",
    }
    needs_motion_file = args.task_variant in {"default", "recovery", "low_freq", "wo_state_estimation"}
    if needs_motion_file and args.motion_file is None:
        parser.error(f"--motion_file is required for task_variant={args.task_variant}")
    if not needs_motion_file and args.motion_file is not None:
        parser.error(f"--motion_file is not used for task_variant={args.task_variant}")

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    train_script = repo_root / "scripts" / "rsl_rl" / "train.py"

    cmd = [
        sys.executable,
        str(train_script),
        "--task",
        task_map[args.task_variant],
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.max_iterations),
        "--device",
        args.device,
        "--run_name",
        args.run_name,
        "--logger",
        args.logger,
        "--log_project_name",
        args.log_project_name,
    ]

    if args.motion_file is not None:
        cmd.extend(["--motion_file", args.motion_file])
    if args.getup_target_json is not None:
        if needs_motion_file:
            parser.error("--getup_target_json is only used for direct get-up task variants")
        cmd.extend(["--getup_target_json", args.getup_target_json])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.video:
        cmd.extend(["--video", "--video_length", str(args.video_length), "--video_interval", str(args.video_interval)])
    if args.headless:
        cmd.append("--headless")
    if args.enable_cameras:
        cmd.append("--enable_cameras")

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)

    print("[INFO] Launching T800 training command:")
    print(" ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
