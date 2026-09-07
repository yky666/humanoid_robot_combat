#!/usr/bin/env python3
"""Stage an accepted T800 recovery policy for isolated Native SDK integration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml


WBT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WBT_ROOT / "source" / "whole_body_tracking"))

from whole_body_tracking.robots.t800_joint_order import (  # noqa: E402
    T800_POLICY_JOINT_NAMES,
    T800_SDK_POLICY_JOINT_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orientation", choices=("prone", "supine"), required=True)
    parser.add_argument("--policy-onnx", type=Path, required=True, help="Actor-only ONNX policy")
    parser.add_argument("--policy-mnn", type=Path, required=True, help="MNN 2.9.5 conversion of the actor")
    parser.add_argument("--trajectory", type=Path, required=True, help="Validated tracking NPZ")
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--rollout-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tag", default="rl_recovery_candidate")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_passed_report(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise ValueError(f"{label} has not passed: {path}")
    return payload


def tensor_shape(value_info: Any) -> list[int | str]:
    return [
        dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def metadata_values(metadata: dict[str, str], key: str, *, numeric: bool = False) -> list:
    if key not in metadata:
        raise ValueError(f"ONNX metadata is missing {key!r}")
    values = [value.strip() for value in metadata[key].split(",") if value.strip()]
    return [float(value) for value in values] if numeric else values


def validate_actor(path: Path) -> dict:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("staging requires the Python onnx package") from error

    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    inputs = {item.name: tensor_shape(item) for item in model.graph.input}
    outputs = {item.name: tensor_shape(item) for item in model.graph.output}
    if inputs != {"obs": [1, 140]} or outputs != {"actions": [1, 25]}:
        raise ValueError(f"unexpected actor contract: inputs={inputs}, outputs={outputs}")

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    if metadata_values(metadata, "joint_names") != T800_POLICY_JOINT_NAMES:
        raise ValueError("ONNX joint_names is not canonical T800 policy order")
    if metadata_values(metadata, "trajectory_joint_names") != T800_POLICY_JOINT_NAMES:
        raise ValueError("ONNX trajectory_joint_names is not canonical T800 policy order")
    checkpoint_sha256 = metadata.get("checkpoint_sha256")
    if not checkpoint_sha256:
        raise ValueError("ONNX metadata is missing checkpoint_sha256")
    output = {
        "default_joint_pos": metadata_values(metadata, "default_joint_pos", numeric=True),
        "joint_stiffness": metadata_values(metadata, "joint_stiffness", numeric=True),
        "joint_damping": metadata_values(metadata, "joint_damping", numeric=True),
        "observation_names": metadata_values(metadata, "observation_names"),
        "observation_history_lengths": [
            int(float(value)) for value in metadata_values(metadata, "observation_history_lengths")
        ],
        "action_scale": metadata_values(metadata, "action_scale", numeric=True),
        "checkpoint_sha256": checkpoint_sha256,
    }
    for key in ("default_joint_pos", "joint_stiffness", "joint_damping", "action_scale"):
        if len(output[key]) != 25:
            raise ValueError(f"ONNX metadata {key} has {len(output[key])} values, expected 25")
    return output


def require_hash_match(report: dict, report_key: str, asset: Path, label: str) -> None:
    expected = report.get(report_key)
    actual = sha256(asset)
    if expected != actual:
        raise ValueError(f"{label} hash mismatch: report={expected!r}, actual={actual}")


def main() -> int:
    args = parse_args()
    onnx_path = args.policy_onnx.expanduser().resolve()
    mnn_path = args.policy_mnn.expanduser().resolve()
    trajectory_path = args.trajectory.expanduser().resolve()
    reference_report_path = args.reference_report.expanduser().resolve()
    rollout_report_path = args.rollout_report.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    assets = (onnx_path, mnn_path, trajectory_path, reference_report_path, rollout_report_path)
    missing = [str(path) for path in assets if not path.is_file()]
    if missing:
        raise SystemExit(f"missing input files: {missing}")

    reference_report = load_passed_report(reference_report_path, "reference report")
    rollout_report = load_passed_report(rollout_report_path, "rollout report")
    if reference_report.get("orientation") != args.orientation:
        raise SystemExit(
            f"reference orientation={reference_report.get('orientation')!r}, expected {args.orientation!r}"
        )
    require_hash_match(reference_report, "motion_sha256", trajectory_path, "trajectory")
    total_trials = int(rollout_report.get("total_trials", 0))
    success_rate = float(rollout_report.get("success_rate", 0.0))
    if total_trials < 320 or success_rate < 0.95:
        raise SystemExit(f"rollout gate requires >=320 trials and >=0.95 success: {total_trials=}, {success_rate=}")

    metadata = validate_actor(onnx_path)
    rollout_checkpoint_sha256 = rollout_report.get("checkpoint_sha256")
    if metadata.pop("checkpoint_sha256") != rollout_checkpoint_sha256:
        raise SystemExit(
            "ONNX checkpoint_sha256 does not match the checkpoint evaluated by the rollout report"
        )
    state_name = f"qualifier_recovery_{args.orientation}"
    package_root = output_root / args.tag
    if package_root.exists() and any(package_root.iterdir()) and not args.force:
        raise SystemExit(f"output exists; pass --force to replace files: {package_root}")

    policies = package_root / "policies"
    trajectories = package_root / "trajectories"
    reports = package_root / "reports"
    for directory in (policies, trajectories, reports):
        directory.mkdir(parents=True, exist_ok=True)
    destinations = {
        "policy_onnx": policies / f"{state_name}.onnx",
        "policy_mnn": policies / f"{state_name}.mnn",
        "trajectory": trajectories / f"{state_name}_tracking.npz",
        "reference_report": reports / f"{state_name}_reference.json",
        "rollout_report": reports / f"{state_name}_rollout.json",
    }
    for source, destination in zip(assets, destinations.values(), strict=True):
        shutil.copy2(source, destination)

    config = {
        "policy_file": f"{args.tag}/policies/{destinations['policy_mnn'].name}",
        "trajectory_file_npz": f"{args.tag}/trajectories/{destinations['trajectory'].name}",
        **metadata,
        "joint_names": T800_SDK_POLICY_JOINT_NAMES,
        "resident_control": True,
        "action_clip": 1.0,
        "transition_time": 0.3,
        "max_initial_pose_error": 0.12,
        "joint_limit_margin": 0.02,
        "expected_observation_dim": 140,
    }
    config_path = package_root / f"{state_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest_files = {
        name: {"path": str(path.relative_to(package_root)), "sha256": sha256(path)}
        for name, path in destinations.items()
    }
    manifest_files["config"] = {"path": config_path.name, "sha256": sha256(config_path)}
    manifest = {
        "status": "staged_not_hardware_approved",
        "state": state_name,
        "orientation": args.orientation,
        "rollout_trials": total_trials,
        "rollout_success_rate": success_rate,
        "files": manifest_files,
        "integration_gate": "add to an isolated recovery-only FSM before any production graph",
    }
    manifest_path = package_root / "DEPLOYMENT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
