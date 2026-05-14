#!/usr/bin/env python3
"""Launch a batch of T800 smoke or formal training runs from the combat manifest."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys


def load_manifest(manifest_path: pathlib.Path) -> list[dict]:
    data = json.loads(manifest_path.read_text())
    return data["motions"]


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Batch-launch T800 smoke or formal training runs.")
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=repo_root / "configs" / "t800_combat_motion_manifest.json",
        help="Path to the motion manifest JSON.",
    )
    parser.add_argument("--stage", choices=("smoke", "formal"), default="smoke", help="Training stage preset to use.")
    parser.add_argument("--groups", nargs="*", default=None, help="Optional motion groups to include.")
    parser.add_argument("--motions", nargs="*", default=None, help="Optional motion names to include.")
    parser.add_argument("--device", default="cuda:0", help="Torch device passed into train_t800.py.")
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--logger", default="wandb", help="Logger backend.")
    parser.add_argument("--project", default="t800_boxing", help="W&B project name.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override num_envs for all selected motions.")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Override max_iterations for all selected motions.",
    )
    parser.add_argument(
        "--resume_load_run",
        default="2026-04-17_12-51-22_cmu13_18_t800_v2_from1499",
        help="Formal-stage resume source run.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        default="model_29999.pt",
        help="Formal-stage resume checkpoint.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest.resolve())
    trainer = repo_root / "scripts" / "rsl_rl" / "train_t800.py"

    default_num_envs = 64 if args.stage == "smoke" else 1024
    default_iterations = 1 if args.stage == "smoke" else 30000

    selected = []
    for motion in manifest:
        if args.groups and motion["group"] not in args.groups:
            continue
        if args.motions and motion["name"] not in args.motions:
            continue
        selected.append(motion)

    if not selected:
        print("[WARN] No motions matched the requested filters.")
        return 1

    env = None
    if args.gpu is not None:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for motion in selected:
        run_name = motion["smoke_run_name"] if args.stage == "smoke" else motion["formal_run_name"]
        cmd = [
            sys.executable,
            str(trainer),
            "--motion_file",
            motion["output_path"],
            "--num_envs",
            str(args.num_envs or default_num_envs),
            "--max_iterations",
            str(args.max_iterations or default_iterations),
            "--headless",
            "--device",
            args.device,
            "--run_name",
            run_name,
            "--logger",
            args.logger,
            "--log_project_name",
            args.project,
        ]
        if args.stage == "formal":
            cmd.extend(
                [
                    "--",
                    "--resume",
                    "true",
                    "--load_run",
                    args.resume_load_run,
                    "--checkpoint",
                    args.resume_checkpoint,
                ]
            )

        print(f"[RUN ] {motion['name']} -> {run_name}")
        print("       " + " ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, check=False, env=env)
        if result.returncode != 0:
            print(f"[FAIL] {motion['name']} exited with code {result.returncode}")
            return result.returncode

    print("[OK] Finished launching selected runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
