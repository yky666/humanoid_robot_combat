#!/usr/bin/env python3
"""Evaluate a T800 policy from motion phase zero and write a stability report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion_file", required=True, help="One canonical T800 tracking NPZ.")
parser.add_argument("--output", type=Path, required=True, help="JSON report path.")
parser.add_argument("--episodes", type=int, default=5, help="Evaluation batches per environment.")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments per batch.")
parser.add_argument("--min_success_rate", type=float, default=0.95)
parser.add_argument("--task", default="Tracking-Flat-T800-v0")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# ``hydra_task_config`` intentionally discards the decorated function's return
# value.  Keep the physical gate result outside the decorator so the process
# exit code still represents the report status for the staged shell queue.
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
from whole_body_tracking.utils.rsl_rl_compat import adapt_legacy_ppo_cfg


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
) -> bool:
    global EVALUATION_PASSED
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.load_run = Path(agent_cfg.load_run).name
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = str(Path(args_cli.motion_file).expanduser().resolve())
    env_cfg.commands.motion.motion_start_reset_ratio = 1.0
    env_cfg.commands.motion.debug_vis = False
    if hasattr(env_cfg.scene, "contact_forces"):
        env_cfg.scene.contact_forces.debug_vis = False

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Evaluating checkpoint: {checkpoint_path}")
    print(f"[INFO] Motion: {env_cfg.commands.motion.motion_file}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    if isinstance(raw_env, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        raw_env = env.unwrapped
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, adapt_legacy_ppo_cfg(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=raw_env.device)
    command = raw_env.command_manager.get_term("motion")
    if len(command.motion.motion_files) != 1:
        raise ValueError("Evaluation requires exactly one motion NPZ.")
    horizon = int(command.motion.traj_lengths[0].item()) - 1
    if horizon <= 0:
        raise ValueError(f"Motion has invalid horizon: {horizon}")
    if horizon >= raw_env.max_episode_length:
        raise ValueError(
            f"Motion horizon {horizon} must be shorter than max episode length {raw_env.max_episode_length}."
        )

    successes = 0
    total_trials = args_cli.num_envs * args_cli.episodes
    reason_counts = {name: 0 for name in raw_env.termination_manager.active_terms}
    metric_sums = {name: 0.0 for name in command.metrics}
    metric_samples = 0

    for episode in range(args_cli.episodes):
        obs, _ = env.reset()
        command.refresh_relative_body_targets()
        obs = env.get_observations()
        failed = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=raw_env.device)
        for _ in range(horizon):
            # Keep simulator state updates outside inference mode.  IsaacLab resets
            # command metrics in-place between batches, which inference tensors forbid.
            with torch.no_grad():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            current_dones = dones.bool()
            first_failures = current_dones & ~failed
            if torch.any(first_failures):
                for name in reason_counts:
                    term = raw_env.termination_manager.get_term(name)
                    reason_counts[name] += int(torch.count_nonzero(term & first_failures).item())
            failed |= current_dones
            for name, value in command.metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value.mean().item())
            metric_samples += 1
        batch_successes = int(torch.count_nonzero(~failed).item())
        successes += batch_successes
        print(
            f"[EVAL] batch={episode + 1}/{args_cli.episodes} "
            f"success={batch_successes}/{args_cli.num_envs}"
        )

    success_rate = successes / max(total_trials, 1)
    passed = success_rate >= args_cli.min_success_rate
    report = {
        "status": "passed" if passed else "failed",
        "motion_file": env_cfg.commands.motion.motion_file,
        "checkpoint": checkpoint_path,
        "joint_order": command.motion.joint_names,
        "horizon_steps": horizon,
        "episodes": args_cli.episodes,
        "num_envs": args_cli.num_envs,
        "total_trials": total_trials,
        "successes": successes,
        "success_rate": success_rate,
        "min_success_rate": args_cli.min_success_rate,
        "first_failure_reasons": reason_counts,
        "mean_metrics": {
            name: value / max(metric_samples, 1) for name, value in metric_sums.items()
        },
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
    # Isaac Sim's shutdown path exits the interpreter with code 0.  Flush the
    # JSON/report logs and terminate explicitly so a failed physical gate can
    # never be mistaken for a passed policy by the shell queue.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
