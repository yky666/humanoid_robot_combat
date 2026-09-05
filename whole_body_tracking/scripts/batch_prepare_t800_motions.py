#!/usr/bin/env python3
"""Batch-convert combat motions into tracking-ready T800 NPZ files."""

from __future__ import annotations

import argparse
import json
import numpy as np
import pathlib
import selectors
import subprocess
import sys
import time


REQUIRED_NPZ_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def load_manifest(manifest_path: pathlib.Path) -> list[dict]:
    data = json.loads(manifest_path.read_text())
    return data["motions"]


def validate_npz(path: pathlib.Path) -> bool:
    try:
        with np.load(path) as data:
            if any(key not in data for key in REQUIRED_NPZ_KEYS):
                return False
            return data["joint_pos"].ndim == 2 and data["body_pos_w"].ndim == 3
    except Exception:
        return False


def stop_process(process: subprocess.Popen, grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_converter(
    cmd: list[str],
    output_path: pathlib.Path,
    saved_exit_timeout: float,
    process_timeout: float,
) -> int:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    start_time = time.monotonic()
    saved_time = None

    while True:
        for key, _ in selector.select(timeout=1.0):
            line = key.fileobj.readline()
            if line:
                print(line, end="", flush=True)

        if saved_time is None and validate_npz(output_path):
            saved_time = time.monotonic()

        returncode = process.poll()
        if returncode is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
            return 0 if validate_npz(output_path) else returncode

        now = time.monotonic()
        if saved_time is not None and now - saved_time > saved_exit_timeout:
            print(
                f"[WARN] Output is valid but converter is still running after {saved_exit_timeout:.1f}s; "
                "terminating child process.",
                flush=True,
            )
            stop_process(process, grace_seconds=10.0)
            return 0 if validate_npz(output_path) else 1

        if process_timeout > 0 and now - start_time > process_timeout:
            print(f"[FAIL] Converter timed out after {process_timeout:.1f}s.", flush=True)
            stop_process(process, grace_seconds=10.0)
            return 0 if validate_npz(output_path) else 124


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
    parser.add_argument(
        "--saved_exit_timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for Isaac Sim to exit after a valid output NPZ is detected.",
    )
    parser.add_argument(
        "--process_timeout",
        type=float,
        default=900.0,
        help="Maximum seconds to wait for one converter subprocess. Set <=0 to disable.",
    )
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
            if validate_npz(output_path):
                print(f"[SKIP] {motion['name']}: {output_path} already exists")
                continue
            print(f"[WARN] {motion['name']}: {output_path} exists but is invalid; rebuilding")

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
        returncode = run_converter(cmd, output_path, args.saved_exit_timeout, args.process_timeout)
        if returncode != 0:
            print(f"[FAIL] {motion['name']} exited with code {returncode}")
            return returncode

    print("[OK] Finished preparing selected motions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
