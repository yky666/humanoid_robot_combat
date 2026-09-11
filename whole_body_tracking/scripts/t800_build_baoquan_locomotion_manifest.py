#!/usr/bin/env python3
"""Build a cleaned manifest for future T800 baoquan locomotion training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=project_root / "results" / "t800_baoquan_locomotion_20260910" / "dataset_audit.json",
        help="Audit JSON produced by t800_audit_baoquan_locomotion_dataset.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "results" / "t800_baoquan_locomotion_20260910" / "command_manifest.json",
        help="Cleaned manifest output path.",
    )
    return parser.parse_args()


def build_manifest(audit: dict) -> dict:
    usable = [item for item in audit["files"] if item["quality"] == "usable"]
    rejected = [item for item in audit["files"] if item["quality"] != "usable"]
    by_command: dict[str, list[dict]] = {}
    for item in usable:
        by_command.setdefault(item["command_hint"], []).append(item)

    missing_commands = [
        command
        for command in ("forward", "backward", "left_strafe", "right_strafe")
        if command not in by_command
    ]

    entries = []
    for item in usable:
        entries.append(
            {
                "relative_path": item["relative_path"],
                "path": item["path"],
                "command_hint": item["command_hint"],
                "measured_mean_velocity_xy": item["mean_velocity_xy"],
                "dominant_world_axis": item["dominant_world_axis"],
                "duration_s": item["duration_s"],
                "frames": item["frames"],
                "baoquan_upper_l2": item["baoquan_upper_l2"],
                "use_as": "lower_body_style_and_command_seed",
            }
        )

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_audit": audit.get("dataset_root"),
        "baoquan_target": audit.get("baoquan_target"),
        "policy": {
            "final_locomotion_rule": "Do not use EngineAI official loco. Train a custom command-conditioned policy.",
            "upper_body_target": "Hold measured baoquan/boxing guard during all locomotion commands.",
            "command_space": ["vx", "vy", "yaw_rate"],
            "dataset_role": "The cleaned URKL movement clips are style/seed data, not an official loco policy.",
        },
        "entries": entries,
        "rejected": [
            {
                "relative_path": item["relative_path"],
                "command_hint": item["command_hint"],
                "reasons": item["reasons"],
            }
            for item in rejected
        ],
        "missing_commands": missing_commands,
        "recommended_augmentation": [
            "Re-record right_strafe, or implement a verified left/right joint mirror before using mirrored data.",
            "Do not synthesize right_strafe by simply negating world x without checking joint symmetry and heading frame.",
            "Treat forward/backward labels cautiously because current world-frame displacement is similar; use robot-frame heading alignment in the locomotion env.",
        ],
    }


def write_markdown(manifest: dict, path: Path) -> None:
    lines = [
        "# T800 Baoquan Locomotion Command Manifest",
        "",
        f"- Created at: `{manifest['created_at']}`",
        f"- Baoquan target: `{manifest['baoquan_target']}`",
        f"- Missing commands: `{', '.join(manifest['missing_commands']) or 'none'}`",
        "",
        "| file | command | seconds | measured mean vxy | axis | baoquan upper L2 | role |",
        "|---|---|---:|---|---|---:|---|",
    ]
    for item in manifest["entries"]:
        vxy = f"{item['measured_mean_velocity_xy'][0]:.3f}, {item['measured_mean_velocity_xy'][1]:.3f}"
        l2 = "" if item["baoquan_upper_l2"] is None else f"{item['baoquan_upper_l2']:.3f}"
        lines.append(
            f"| `{item['relative_path']}` | {item['command_hint']} | {item['duration_s']:.2f} | "
            f"{vxy} | {item['dominant_world_axis']} | {l2} | {item['use_as']} |"
        )
    lines.extend(["", "## Rejected"])
    for item in manifest["rejected"]:
        lines.append(f"- `{item['relative_path']}` ({item['command_hint']}): {'; '.join(item['reasons'])}")
    lines.extend(["", "## Recommended Augmentation"])
    for item in manifest["recommended_augmentation"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    manifest = build_manifest(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(manifest, args.output.with_suffix(".md"))
    print(f"[OK] wrote {args.output}")
    print(f"[INFO] entries={len(manifest['entries'])} rejected={len(manifest['rejected'])}")
    if manifest["missing_commands"]:
        print(f"[WARN] missing commands: {', '.join(manifest['missing_commands'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
