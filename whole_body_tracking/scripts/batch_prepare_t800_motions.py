#!/usr/bin/env python3
"""Batch-convert combat motions into tracking-ready T800 NPZ files."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


def load_manifest(manifest_path: pathlib.Path) -> list[dict]:
    data = json.loads(manifest_path.read_text())
    return data["motions"]


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Batch-convert T800 combat motions into tracking-ready NPZ files.")
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=repo_root / "configs" / "t800_combat_motion_manifest.json",
        help="Path to the motion manifest JSON.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional motion groups to include, e.g. accad_combat_batch legacy_local.",
    )
    parser.add_argument(
        "--motions",
        nargs="*",
        default=None,
        help="Optional motion names to include.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even if the target NPZ already exists.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest.resolve())
    converter = repo_root / "scripts" / "t800_csv_to_npz.py"

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

    for motion in selected:
        output_path = pathlib.Path(motion["output_path"])
        if output_path.exists() and not args.force:
            print(f"[SKIP] {motion['name']}: {output_path} already exists")
            continue

        cmd = [
            sys.executable,
            str(converter),
            "--input_file",
            motion["input_file"],
            "--input_format",
            motion["input_format"],
            "--input_fps",
            str(motion["input_fps"]),
            "--output_name",
            output_path.stem,
            "--output_path",
            str(output_path),
            "--output_fps",
            "50",
            "--skip_wandb_upload",
            "--headless",
        ]
        print(f"[RUN ] {motion['name']}")
        print("       " + " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[FAIL] {motion['name']} exited with code {result.returncode}")
            return result.returncode

    print("[OK] Finished preparing selected motions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
