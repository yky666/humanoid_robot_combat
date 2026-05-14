#!/usr/bin/env python3
"""Run staged T800 tracking training with checkpoint hand-off."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


PHASES = ("clean", "noise", "delay", "push")


def _find_latest_run(log_root: pathlib.Path, run_name: str) -> pathlib.Path:
    candidates = sorted(log_root.glob(f"*_{run_name}"))
    if not candidates:
        raise FileNotFoundError(f"No run directory found for {run_name!r} under {log_root}")
    return candidates[-1]


def _find_latest_checkpoint(run_dir: pathlib.Path) -> pathlib.Path:
    checkpoints = sorted(run_dir.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return checkpoints[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train T800 tracking with staged robustness curriculum.")
    parser.add_argument("--motion_file", required=True, help="Tracking-ready T800 motion npz.")
    parser.add_argument("--device", default="cuda:0", help="Torch device passed into train_t800.py.")
    parser.add_argument("--num_envs", type=int, default=1024, help="Parallel environments per phase.")
    parser.add_argument("--max_iterations", type=int, default=30000, help="Iterations per phase.")
    parser.add_argument("--logger", default="wandb", help="Logger backend.")
    parser.add_argument("--log_project_name", default="t800_boxing", help="W&B project name.")
    parser.add_argument("--run_prefix", default="t800_curriculum", help="Prefix used to build phase run names.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without UI.")
    parser.add_argument(
        "--start_phase",
        choices=PHASES,
        default="clean",
        help="Phase to start from. Later phases auto-resume from the prior phase output.",
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    trainer = repo_root / "scripts" / "rsl_rl" / "train_t800.py"
    log_root = repo_root / "logs" / "rsl_rl" / "t800_flat"

    start_idx = PHASES.index(args.start_phase)
    previous_run_name: str | None = None
    previous_checkpoint_name: str | None = None

    for phase in PHASES[start_idx:]:
        run_name = f"{args.run_prefix}_{phase}"
        cmd = [
            sys.executable,
            str(trainer),
            "--motion_file",
            args.motion_file,
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(args.max_iterations),
            "--device",
            args.device,
            "--run_name",
            run_name,
            "--logger",
            args.logger,
            "--log_project_name",
            args.log_project_name,
            "--task_variant",
            phase,
        ]
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        if args.headless:
            cmd.append("--headless")
        if previous_run_name and previous_checkpoint_name:
            cmd.extend(
                [
                    "--",
                    "--resume",
                    "true",
                    "--load_run",
                    previous_run_name,
                    "--checkpoint",
                    previous_checkpoint_name,
                ]
            )

        print(f"[INFO] Starting phase {phase}: {' '.join(cmd)}")
        return_code = subprocess.run(cmd, check=False).returncode
        if return_code != 0:
            return return_code

        run_dir = _find_latest_run(log_root, run_name)
        checkpoint = _find_latest_checkpoint(run_dir)
        previous_run_name = run_dir.name
        previous_checkpoint_name = checkpoint.name
        print(f"[INFO] Phase {phase} complete. Next resume source: {previous_run_name}/{previous_checkpoint_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
