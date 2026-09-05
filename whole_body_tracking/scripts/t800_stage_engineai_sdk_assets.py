#!/usr/bin/env python3
"""Stage accepted canonical T800 policies and trajectories for EngineAI SDK."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import onnx
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "source" / "whole_body_tracking"))

from whole_body_tracking.robots.t800_joint_order import (  # noqa: E402
    T800_POLICY_JOINT_NAMES,
    T800_SDK_POLICY_JOINT_NAMES,
)


MOTION_CONFIG_NAMES = {
    "front_kick": "qualifier_front_kick",
    "spinning_kick": "qualifier_spinning_kick",
    "straight_punch": "qualifier_straight_punch",
    "hook_punch": "qualifier_hook_punch",
    "jab_left": "qualifier_jab_left",
    "recovery_supine": "qualifier_recovery_supine",
}
NATIVE_RECOVERY_BACKEND = "engineai_native_sdk_mnn"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_passed_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"acceptance report is missing: {path}")
    report = load_json(path)
    if report.get("status") != "passed":
        raise RuntimeError(f"acceptance report has not passed: {path}")
    return report


def exported_policy(report: dict) -> Path:
    run_dir = Path(report.get("run_dir") or Path(report["checkpoint"]).parent)
    policy = run_dir / "exported" / "policy.onnx"
    if not policy.is_file():
        raise FileNotFoundError(f"exported policy is missing: {policy}")
    return policy


def require_asset(report: dict, key: str) -> Path:
    path = Path(report[key]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"recovery {key} is missing: {path}")
    expected_hash = report.get(f"{key}_sha256")
    if expected_hash:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"recovery {key} hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
    return path


def copy_asset(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def metadata_list(metadata: dict[str, str], key: str, *, numeric: bool = False) -> list:
    if key not in metadata:
        raise KeyError(f"ONNX metadata is missing {key!r}")
    values = [value.strip() for value in metadata[key].split(",") if value.strip()]
    return [float(value) for value in values] if numeric else values


def policy_metadata(policy: Path) -> dict:
    model = onnx.load(policy, load_external_data=False)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    joint_names = metadata_list(metadata, "joint_names")
    trajectory_joint_names = metadata_list(metadata, "trajectory_joint_names")
    if joint_names != T800_POLICY_JOINT_NAMES:
        raise ValueError(f"policy joint order is not canonical: {policy}")
    if trajectory_joint_names != T800_POLICY_JOINT_NAMES:
        raise ValueError(f"trajectory joint order is not canonical: {policy}")
    return {
        "default_joint_pos": metadata_list(metadata, "default_joint_pos", numeric=True),
        "joint_stiffness": metadata_list(metadata, "joint_stiffness", numeric=True),
        "joint_damping": metadata_list(metadata, "joint_damping", numeric=True),
        "observation_names": metadata_list(metadata, "observation_names"),
        "observation_history_lengths": [
            int(float(value)) for value in metadata_list(metadata, "observation_history_lengths")
        ],
        "action_scale": metadata_list(metadata, "action_scale", numeric=True),
    }


def resolve_manifest_paths(manifest_path: Path) -> dict[str, Path]:
    payload = load_json(manifest_path)
    paths = {}
    for motion in payload["motions"]:
        path = Path(motion["motion_file"])
        paths[motion["name"]] = path if path.is_absolute() else (REPO_ROOT / path).resolve()
    return paths


def update_mode_scopes(payload: dict, config_scope: str, task_scope: str) -> None:
    for entries in payload.get("mode", {}).values():
        for entry in entries:
            if entry.get("tag") == "motion_task":
                entry["scope"] = task_scope
            elif entry.get("tag") in MOTION_CONFIG_NAMES.values():
                entry["scope"] = f"{config_scope}/{entry['tag']}"


def integrate_native_recovery_task(payload: dict, official_task_path: Path) -> None:
    official_payload = yaml.safe_load(official_task_path.read_text(encoding="utf-8"))
    matches = [task for task in official_payload["tasks"] if task.get("motion") == "supine_to_stance"]
    if len(matches) != 1:
        raise ValueError(f"expected one official supine_to_stance task in {official_task_path}")

    tasks = [task for task in payload["tasks"] if task.get("motion") != "qualifier_recovery_supine"]
    for task in tasks:
        transitions = task.get("manual_transition")
        if transitions:
            task["manual_transition"] = [
                target for target in transitions if target != "qualifier_recovery_supine"
            ]
    passive = next((task for task in tasks if task.get("motion") == "passive"), None)
    if passive is None:
        raise ValueError("qualifier task graph has no passive state")
    passive.setdefault("manual_transition", [])
    if "supine_to_stance" not in passive["manual_transition"]:
        passive["manual_transition"].append("supine_to_stance")
    tasks.append(matches[0])
    payload["tasks"] = tasks


def integrate_native_recovery_mode(payload: dict) -> None:
    for entries in payload.get("mode", {}).values():
        entries[:] = [entry for entry in entries if entry.get("tag") != "qualifier_recovery_supine"]
        if not any(entry.get("tag") == "rl_supine_to_stance" for entry in entries):
            entries.append({"tag": "rl_supine_to_stance", "scope": "rl_supine_to_stance/default"})


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "t800_qualifier_training_canonical_v1_20260902",
    )
    parser.add_argument("--sdk-root", type=Path, default=REPO_ROOT.parent / "engineai_robotics_native_sdk")
    parser.add_argument(
        "--template-sdk-root",
        type=Path,
        help="SDK checkout containing the qualifier ONNX configuration templates",
    )
    parser.add_argument("--tag", default="rl_qualifier_canonical_v1_20260902")
    args = parser.parse_args()

    state_root = args.state_root.expanduser().resolve()
    sdk_root = args.sdk_root.expanduser().resolve()
    template_sdk_root = (
        args.template_sdk_root.expanduser().resolve() if args.template_sdk_root else sdk_root
    )
    source_config_dir = (
        template_sdk_root / "assets" / "config" / "t800" / "rl_qualifier_approved_20260902"
    )
    if not source_config_dir.is_dir() and args.template_sdk_root is None:
        fallback = REPO_ROOT.parent / "engineai_robotics_native_sdk"
        fallback_config = fallback / "assets" / "config" / "t800" / "rl_qualifier_approved_20260902"
        if fallback_config.is_dir():
            template_sdk_root = fallback.resolve()
            source_config_dir = fallback_config.resolve()
    output_config_dir = sdk_root / "assets" / "config" / "t800" / args.tag
    task_source = (
        template_sdk_root / "assets" / "config" / "t800" / "task_motion" / "qualifier_approved.yaml"
    )
    mode_source = template_sdk_root / "assets" / "config" / "t800" / "mode_qualifier_approved.yaml"

    standing_report = require_passed_report(state_root / "best" / "joint_standing5.json")
    recovery_report = require_passed_report(state_root / "best" / "recovery_supine.json")
    standing_policy = exported_policy(standing_report)
    standing_metadata = policy_metadata(standing_policy)
    native_recovery = recovery_report.get("policy_backend") == NATIVE_RECOVERY_BACKEND
    recovery_policy = None if native_recovery else exported_policy(recovery_report)
    recovery_metadata = None if native_recovery else policy_metadata(recovery_policy)

    transition_paths = resolve_manifest_paths(state_root / "stand_action_stand" / "manifest.json")
    trajectory_paths = {name: transition_paths[name] for name in MOTION_CONFIG_NAMES if name != "recovery_supine"}
    if not native_recovery:
        source_paths = resolve_manifest_paths(state_root / "source_manifest_resolved.json")
        trajectory_paths["recovery_supine"] = source_paths["recovery_supine"]

    policy_destinations = {"standing": output_config_dir / "policies" / "standing5" / "policy.onnx"}
    copy_asset(standing_policy, policy_destinations["standing"])
    if native_recovery:
        native_root = sdk_root / "assets" / "config" / "t800" / "rl_supine_to_stance"
        policy_destinations["recovery"] = copy_asset(
            require_asset(recovery_report, "policy"),
            native_root / "policy" / "T800_supine_to_stance.mnn",
        )
        copy_asset(
            require_asset(recovery_report, "trajectory"),
            native_root / "trajectory" / "T800_supine_to_stance.npy",
        )
        copy_asset(require_asset(recovery_report, "config"), native_root / "default.yaml")
    else:
        policy_destinations["recovery"] = output_config_dir / "policies" / "recovery_supine" / "policy.onnx"
        copy_asset(recovery_policy, policy_destinations["recovery"])

    for motion_name, source_trajectory in trajectory_paths.items():
        if not source_trajectory.is_file():
            raise FileNotFoundError(f"trajectory is missing: {source_trajectory}")
        config_name = MOTION_CONFIG_NAMES[motion_name]
        trajectory_destination = output_config_dir / "trajectories" / f"{motion_name}_tracking.npz"
        trajectory_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_trajectory, trajectory_destination)

        template = yaml.safe_load((source_config_dir / f"{config_name}.yaml").read_text(encoding="utf-8"))
        metadata = recovery_metadata if motion_name == "recovery_supine" else standing_metadata
        policy_relative = (
            f"{args.tag}/policies/recovery_supine/policy.onnx"
            if motion_name == "recovery_supine"
            else f"{args.tag}/policies/standing5/policy.onnx"
        )
        template.update(metadata)
        template["policy_file"] = policy_relative
        template["trajectory_file_npz"] = f"{args.tag}/trajectories/{motion_name}_tracking.npz"
        template["joint_names"] = T800_SDK_POLICY_JOINT_NAMES
        template["resident_control"] = True
        write_yaml(output_config_dir / f"{config_name}.yaml", template)

    task_destination = sdk_root / "assets" / "config" / "t800" / "task_motion" / "qualifier_canonical_v1.yaml"
    task_payload = yaml.safe_load(task_source.read_text(encoding="utf-8"))
    if native_recovery:
        integrate_native_recovery_task(task_payload, require_asset(recovery_report, "task_config"))
    write_yaml(task_destination, task_payload)

    mode_payload = yaml.safe_load(mode_source.read_text(encoding="utf-8"))
    update_mode_scopes(mode_payload, args.tag, "task_motion/qualifier_canonical_v1")
    if native_recovery:
        integrate_native_recovery_mode(mode_payload)
    mode_destination = sdk_root / "assets" / "config" / "t800" / "mode_qualifier_canonical_v1.yaml"
    write_yaml(mode_destination, mode_payload)

    ready = {
        "status": "passed",
        "joint_order": "t800_policy_v1",
        "standing_policy": str(policy_destinations["standing"]),
        "recovery_policy": str(policy_destinations["recovery"]),
        "recovery_backend": recovery_report.get("policy_backend", "isaaclab_onnx"),
        "recovery_task": "supine_to_stance" if native_recovery else "qualifier_recovery_supine",
        "standing_report": str(state_root / "best" / "joint_standing5.json"),
        "recovery_report": str(state_root / "best" / "recovery_supine.json"),
        "mode_config": str(mode_destination),
    }
    (output_config_dir / "READY.json").write_text(
        json.dumps(ready, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(ready, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
