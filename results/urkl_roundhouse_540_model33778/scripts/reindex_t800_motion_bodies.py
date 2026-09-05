#!/usr/bin/env python3
"""Reindex a legacy T800 tracking NPZ to a selected T800 body contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


CANONICAL_34_BODY_NAMES = [
    "LINK_BASE",
    "LINK_HIP_PITCH_L",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_YAW_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_L_TOE",
    "LINK_ANKLE_ROLL_L_HEEL",
    "LINK_HIP_PITCH_R",
    "LINK_HIP_ROLL_R",
    "LINK_HIP_YAW_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_ANKLE_ROLL_R_TOE",
    "LINK_ANKLE_ROLL_R_HEEL",
    "LINK_TORSO_YAW",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    "LINK_WRIST_PITCH_L",
    "LINK_WRIST_ROLL_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_SHOULDER_ROLL_R",
    "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
    "LINK_WRIST_PITCH_R",
    "LINK_WRIST_ROLL_R",
    "LINK_HEAD_PITCH",
    "LINK_HEAD_YAW",
]

# IsaacLab's articulation body order is breadth-first and differs from the
# depth-first order stored in the imported motion. MotionLoader indexes body
# arrays by these runtime indices, so this exact ordering is required for eval.
RUNTIME_30_BODY_NAMES = [
    "LINK_BASE",
    "LINK_HIP_PITCH_L",
    "LINK_HIP_PITCH_R",
    "LINK_TORSO_YAW",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_ROLL_R",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_HEAD_PITCH",
    "LINK_HIP_YAW_L",
    "LINK_HIP_YAW_R",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_ROLL_R",
    "LINK_HEAD_YAW",
    "LINK_KNEE_PITCH_L",
    "LINK_KNEE_PITCH_R",
    "LINK_SHOULDER_YAW_L",
    "LINK_SHOULDER_YAW_R",
    "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_PITCH_R",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_PITCH_R",
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
    "LINK_ELBOW_YAW_L",
    "LINK_ELBOW_YAW_R",
    "LINK_FOOT_L",
    "LINK_FOOT_R",
    "LINK_WRIST_END_L",
    "LINK_WRIST_END_R",
]

TARGETS = {
    "canonical34": (CANONICAL_34_BODY_NAMES, "t800_canonical_34_v1"),
    "runtime30": (RUNTIME_30_BODY_NAMES, "t800_isaaclab_runtime_30_v1"),
}

BODY_ALIASES = {
    "LINK_ANKLE_ROLL_L_TOE": "LINK_FOOT_L",
    "LINK_ANKLE_ROLL_L_HEEL": "LINK_FOOT_L",
    "LINK_ANKLE_ROLL_R_TOE": "LINK_FOOT_R",
    "LINK_ANKLE_ROLL_R_HEEL": "LINK_FOOT_R",
    "LINK_WRIST_PITCH_L": "LINK_WRIST_END_L",
    "LINK_WRIST_ROLL_L": "LINK_WRIST_END_L",
    "LINK_WRIST_PITCH_R": "LINK_WRIST_END_R",
    "LINK_WRIST_ROLL_R": "LINK_WRIST_END_R",
}

BODY_ARRAY_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def decode_names(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1).tolist()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, default="runtime30")
    args = parser.parse_args()

    target_body_names, target_version = TARGETS[args.target]

    with np.load(args.input, allow_pickle=True) as source:
        payload = {key: np.array(source[key]) for key in source.files}

    source_names = decode_names(payload["body_names"])
    source_index = {name: index for index, name in enumerate(source_names)}
    selected_names = [name if name in source_index else BODY_ALIASES.get(name, "") for name in target_body_names]
    missing = [target for target, selected in zip(target_body_names, selected_names) if selected not in source_index]
    if missing:
        raise ValueError(f"Cannot map canonical body names: {missing}")

    indices = [source_index[name] for name in selected_names]
    for key in BODY_ARRAY_KEYS:
        values = payload[key]
        if values.ndim != 3 or values.shape[1] != len(source_names):
            raise ValueError(f"{key} shape {values.shape} does not match {len(source_names)} source bodies")
        payload[key] = values[:, indices]

    payload["body_names"] = np.asarray(target_body_names)
    payload["body_order_version"] = np.asarray(target_version)
    payload["body_order_source"] = np.asarray(str(args.input.resolve()))
    payload["body_order_aliases"] = np.asarray(
        [f"{target}={source}" for target, source in BODY_ALIASES.items()]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(
        f"reindexed={args.output.resolve()} frames={payload['joint_pos'].shape[0]} "
        f"bodies={len(target_body_names)} target={args.target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
