#!/usr/bin/env python3
"""Evaluate a reference-free T800 get-up policy and write a qualification report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True, help="JSON report path.")
parser.add_argument("--episodes", type=int, default=5, help="Evaluation batches per environment.")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments per batch.")
parser.add_argument("--min_success_rate", type=float, default=0.95)
parser.add_argument("--task", default="Getup-Direct-T800-Supine-v0")
parser.add_argument("--target_height", type=float, default=0.72)
parser.add_argument("--max_tilt_rad", type=float, default=0.35)
parser.add_argument("--max_height_error", type=float, default=0.16)
parser.add_argument("--max_joint_error", type=float, default=0.45)
parser.add_argument("--max_root_speed", type=float, default=0.75)
parser.add_argument("--getup_target_json", type=str, default=None, help="Measured target_joint_pos JSON for direct get-up tasks.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "source" / "whole_body_tracking"
sys.path.insert(0, str(SOURCE_ROOT))

EVALUATION_PASSED: bool | None = None

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from t800_getup_target import apply_getup_target_json
from whole_body_tracking.utils.rsl_rl_compat import adapt_legacy_ppo_cfg


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_success(raw_env, target_joint_pos: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    robot = raw_env.scene["robot"]
    target = target_joint_pos.to(robot.device)
    joint_ids = raw_env.action_manager.get_term("joint_pos")._joint_ids
    joint_error = torch.max(torch.abs(robot.data.joint_pos[:, joint_ids] - target.unsqueeze(0)), dim=-1).values
    gravity_z = torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(gravity_z)
    env_origins = raw_env.scene.env_origins.to(robot.data.root_pos_w.device)
    height_error = torch.abs(robot.data.root_pos_w[:, 2] - env_origins[:, 2] - args_cli.target_height)
    root_speed = torch.linalg.norm(robot.data.root_lin_vel_b, dim=-1)
    success = (
        (tilt < args_cli.max_tilt_rad)
        & (height_error < args_cli.max_height_error)
        & (joint_error < args_cli.max_joint_error)
        & (root_speed < args_cli.max_root_speed)
    )
    metrics = {
        "tilt_rad": tilt,
        "height_error_m": height_error,
        "joint_error_rad": joint_error,
        "root_speed_mps": root_speed,
    }
    return success, metrics


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
) -> bool:
    global EVALUATION_PASSED
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.load_run = Path(agent_cfg.load_run).name
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg.scene, "contact_forces"):
        env_cfg.scene.contact_forces.debug_vis = False
    target_source = "env_cfg.target_joint_pos"
    if args_cli.getup_target_json:
        target_source = apply_getup_target_json(env_cfg, args_cli.getup_target_json)
        print(f"[INFO] Using T800 get-up target from: {target_source}")

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Evaluating checkpoint: {checkpoint_path}")
    print(f"[INFO] Task: {args_cli.task}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    if isinstance(raw_env, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        raw_env = env.unwrapped
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, adapt_legacy_ppo_cfg(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=raw_env.device)
    target_joint_pos = torch.tensor(env_cfg.target_joint_pos, dtype=torch.float32, device=raw_env.device)

    successes = 0
    total_trials = args_cli.num_envs * args_cli.episodes
    failure_counts = {"terminated": 0, "time_out": 0, "final_pose": 0}
    metric_sums = {"tilt_rad": 0.0, "height_error_m": 0.0, "joint_error_rad": 0.0, "root_speed_mps": 0.0}
    horizon = int(raw_env.max_episode_length)

    for episode in range(args_cli.episodes):
        obs, _ = env.reset()
        failed = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=raw_env.device)
        for _ in range(horizon):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(raw_env, "termination_manager"):
                terminated = raw_env.termination_manager.terminated.bool()
                time_outs = raw_env.termination_manager.time_outs.bool()
            else:
                terminated = dones.bool()
                time_outs = torch.zeros_like(terminated)
            failed |= terminated
            failure_counts["time_out"] += int(torch.count_nonzero(time_outs).item())
        final_success, metrics = compute_success(raw_env, target_joint_pos)
        final_success &= ~failed
        failure_counts["terminated"] += int(torch.count_nonzero(failed).item())
        failure_counts["final_pose"] += int(torch.count_nonzero(~final_success & ~failed).item())
        successes += int(torch.count_nonzero(final_success).item())
        for name, value in metrics.items():
            metric_sums[name] += float(value.mean().item())
        print(f"[EVAL] batch={episode + 1}/{args_cli.episodes} success={int(torch.count_nonzero(final_success).item())}/{args_cli.num_envs}")

    success_rate = successes / max(total_trials, 1)
    passed = success_rate >= args_cli.min_success_rate
    report = {
        "status": "passed" if passed else "failed",
        "task": args_cli.task,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256(checkpoint_path),
        "target_joint_pos_source": target_source,
        "horizon_steps": horizon,
        "episodes": args_cli.episodes,
        "num_envs": args_cli.num_envs,
        "total_trials": total_trials,
        "successes": successes,
        "success_rate": success_rate,
        "min_success_rate": args_cli.min_success_rate,
        "failure_counts": failure_counts,
        "mean_metrics": {name: value / max(args_cli.episodes, 1) for name, value in metric_sums.items()},
    }
    atomic_write_json(args_cli.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    env.close()
    EVALUATION_PASSED = passed
    return passed


if __name__ == "__main__":
    main()
    exit_code = 0 if EVALUATION_PASSED is True else 2
    print(f"[EVAL] gate_status={EVALUATION_PASSED} exit_code={exit_code}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
