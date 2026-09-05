#!/usr/bin/env python3
"""Migrate legacy T800 tracking NPZ files to the canonical policy joint order."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "whole_body_tracking"
sys.path.insert(0, str(SOURCE_ROOT))

from whole_body_tracking.robots.t800_joint_order import (  # noqa: E402
    T800_JOINT_ORDER_VERSION,
    T800_LEGACY_ISAAC_STORAGE_JOINT_NAMES,
    T800_POLICY_JOINT_NAMES,
    joint_reorder_indices,
)


def decode_names(values: np.ndarray) -> list[str]:
    names = []
    for value in np.asarray(values).reshape(-1).tolist():
        names.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return names


def migrate(input_path: Path, output_path: Path, force: bool) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"Output exists; pass --force to replace it: {output_path}")

    with np.load(input_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}

    source_names = (
        decode_names(arrays["joint_names"])
        if "joint_names" in arrays
        else list(T800_LEGACY_ISAAC_STORAGE_JOINT_NAMES)
    )
    indices = joint_reorder_indices(source_names, T800_POLICY_JOINT_NAMES)
    for key in ("joint_pos", "joint_vel"):
        values = arrays[key]
        if values.ndim != 2 or values.shape[1] != len(source_names):
            raise ValueError(f"{input_path}: {key} has incompatible shape {values.shape}")
        arrays[key] = values[:, indices]

    arrays["joint_names"] = np.asarray(T800_POLICY_JOINT_NAMES)
    arrays["joint_order_version"] = np.asarray(T800_JOINT_ORDER_VERSION)
    arrays["source_joint_names"] = np.asarray(source_names)
    arrays["source_file"] = np.asarray(str(input_path.resolve()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".npz", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        np.savez(temp_path, **arrays)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    with np.load(output_path, allow_pickle=False) as check:
        written_names = decode_names(check["joint_names"])
        if written_names != T800_POLICY_JOINT_NAMES:
            raise RuntimeError(f"Failed to write canonical joint order: {output_path}")
        print(
            f"[OK] {input_path.name} -> {output_path} "
            f"frames={check['joint_pos'].shape[0]} joints={check['joint_pos'].shape[1]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    migrate(args.input.expanduser().resolve(), args.output.expanduser().resolve(), args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
