#!/usr/bin/env python3
"""Gate a T800 policy run using tail-averaged TensorBoard training metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRIC_TAGS = {
    "time_out_rate": "Episode_Termination/time_out",
    "anchor_pos_termination_rate": "Episode_Termination/anchor_pos",
    "anchor_ori_termination_rate": "Episode_Termination/anchor_ori",
    "ee_pos_termination_rate": "Episode_Termination/ee_body_pos",
    "mean_episode_length": "Train/mean_episode_length",
    "joint_pos_l2_error": "Metrics/motion/error_joint_pos",
    "anchor_pos_error": "Metrics/motion/error_anchor_pos",
}


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail", type=int, default=100)
    parser.add_argument("--min-timeout-rate", type=float, default=0.95)
    parser.add_argument("--min-episode-length", type=float, default=450.0)
    parser.add_argument("--max-anchor-pos-termination", type=float, default=0.02)
    parser.add_argument("--max-anchor-ori-termination", type=float, default=0.02)
    parser.add_argument("--max-ee-pos-termination", type=float, default=0.03)
    parser.add_argument("--max-joint-pos-error", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))

    metrics = {}
    missing = []
    last_step = 0
    for name, tag in METRIC_TAGS.items():
        if tag not in scalar_tags:
            missing.append(tag)
            continue
        values = accumulator.Scalars(tag)
        tail_values = values[-max(args.tail, 1) :]
        metrics[name] = float(np.mean([item.value for item in tail_values]))
        if tail_values:
            last_step = max(last_step, int(tail_values[-1].step))

    checks = {
        "time_out_rate": metrics.get("time_out_rate", -np.inf) >= args.min_timeout_rate,
        "mean_episode_length": metrics.get("mean_episode_length", -np.inf) >= args.min_episode_length,
        "anchor_pos_termination_rate": metrics.get("anchor_pos_termination_rate", np.inf)
        <= args.max_anchor_pos_termination,
        "anchor_ori_termination_rate": metrics.get("anchor_ori_termination_rate", np.inf)
        <= args.max_anchor_ori_termination,
        "ee_pos_termination_rate": metrics.get("ee_pos_termination_rate", np.inf)
        <= args.max_ee_pos_termination,
        "joint_pos_l2_error": metrics.get("joint_pos_l2_error", np.inf) <= args.max_joint_pos_error,
    }
    passed = not missing and all(checks.values())
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
    report = {
        "motion": args.motion,
        "status": "passed" if passed else "failed",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "last_step": last_step,
        "tail_samples": args.tail,
        "metrics": metrics,
        "checks": checks,
        "missing_tags": missing,
        "thresholds": {
            "min_time_out_rate": args.min_timeout_rate,
            "min_episode_length": args.min_episode_length,
            "max_anchor_pos_termination": args.max_anchor_pos_termination,
            "max_anchor_ori_termination": args.max_anchor_ori_termination,
            "max_ee_pos_termination": args.max_ee_pos_termination,
            "max_joint_pos_error": args.max_joint_pos_error,
        },
    }
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
