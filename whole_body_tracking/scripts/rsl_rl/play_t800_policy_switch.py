#!/usr/bin/env python3
"""Play a T800 checkpoint while switching among named reference motions."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


DEFAULT_SWITCH_MOTIONS = [
    "MA_Bajiquan_t800",
    "MA_International_Compulsory_Boxing_Routine_I_t800",
    "MA_Punching_Techniques_00001_t800",
    "DAN_breakdance_t800",
    "DAN_running_t800",
]
DEFAULT_MOTION_FILE = "/data2/yangky/test/datasets/MotionDecode_T800_aligned/tracking_manifest_csv.json"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--video", action="store_true", default=False, help="Record a playback video.")
parser.add_argument("--video_length", type=int, default=1500, help="Length of the recorded video in env steps.")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum simulation steps; defaults to video_length.")
parser.add_argument("--video_folder_name", type=str, default="sonic_switch_probe", help="Video subfolder under run/videos.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric USD acceleration.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Tracking-Flat-T800-v0", help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=DEFAULT_MOTION_FILE, help="Path to the T800 motion npz/manifest.")
parser.add_argument(
    "--switch_motions",
    nargs="+",
    default=DEFAULT_SWITCH_MOTIONS,
    help="Motion names or regex patterns to cycle through. Comma-separated values are accepted.",
)
parser.add_argument("--switch_interval", type=int, default=250, help="Steps between reference switches.")
parser.add_argument("--switch_regex", action="store_true", help="Treat switch_motions as regular expressions.")
parser.add_argument(
    "--no_snap_robot_on_switch",
    action="store_true",
    help="Do not reset robot root/joints to the new reference at switch time.",
)
parser.add_argument(
    "--show_debug_vis",
    action="store_true",
    help="Show command/contact debug visualization overlays in playback videos.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.utils.rsl_rl_compat import adapt_legacy_ppo_cfg, get_policy_module


def split_patterns(raw_patterns: list[str]) -> list[str]:
    patterns: list[str] = []
    for item in raw_patterns:
        patterns.extend(part.strip() for part in item.split(",") if part.strip())
    return patterns


def motion_label(path: str) -> str:
    name = pathlib.Path(path).name
    return name.removesuffix("_tracking.npz").removesuffix(".npz")


def match_motion(pattern: str, motion_files: list[str], use_regex: bool) -> int | None:
    if use_regex:
        compiled = re.compile(pattern)
        for index, path in enumerate(motion_files):
            if compiled.search(path) or compiled.search(motion_label(path)):
                return index
        return None

    lowered = pattern.lower()
    for index, path in enumerate(motion_files):
        label = motion_label(path).lower()
        if lowered == label or lowered in label or lowered in path.lower():
            return index
    return None


def build_motion_offsets(command) -> tuple[list[int], list[int], list[str]]:
    lengths_tensor = command.motion.traj_lengths.detach().cpu()
    lengths = [int(item) for item in lengths_tensor.tolist()]
    offsets: list[int] = []
    cursor = 0
    for length in lengths:
        offsets.append(cursor)
        cursor += length
    return offsets, lengths, list(command.motion.motion_files)


def select_switch_motions(command, patterns: list[str], use_regex: bool) -> list[tuple[str, int, int]]:
    offsets, lengths, motion_files = build_motion_offsets(command)
    selected: list[tuple[str, int, int]] = []
    for pattern in patterns:
        index = match_motion(pattern, motion_files, use_regex)
        if index is None:
            print(f"[WARN] Could not find motion pattern: {pattern}")
            continue
        selected.append((motion_label(motion_files[index]), offsets[index], lengths[index]))

    if not selected:
        print("[WARN] No requested switch motions found; falling back to the first motion in the manifest.")
        selected.append((motion_label(motion_files[0]), offsets[0], lengths[0]))
    return selected


def snap_robot_to_reference(raw_env, command) -> None:
    env_ids = torch.arange(raw_env.num_envs, device=raw_env.device)
    root_pos = command.body_pos_w[:, 0].clone()
    root_ori = command.body_quat_w[:, 0].clone()
    root_lin_vel = command.body_lin_vel_w[:, 0].clone()
    root_ang_vel = command.body_ang_vel_w[:, 0].clone()
    joint_pos = command.joint_pos.clone()
    joint_vel = command.joint_vel.clone()

    command.robot.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=command.joint_indexes,
        env_ids=env_ids,
    )
    command.robot.write_root_state_to_sim(
        torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
        env_ids=env_ids,
    )


def apply_switch(raw_env, command, selected: list[tuple[str, int, int]], switch_index: int, snap: bool) -> str:
    label, start, length = selected[switch_index % len(selected)]
    command.time_steps[:] = int(start)
    if snap:
        snap_robot_to_reference(raw_env, command)
    print(f"[SWITCH] {switch_index}: {label} start={start} length={length} snap={snap}")
    return label


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if agent_cfg.load_run is not None:
        normalized_run = pathlib.Path(agent_cfg.load_run).name
        if normalized_run != agent_cfg.load_run:
            print(f"[INFO] Normalized load_run to local run name: {normalized_run}")
            agent_cfg.load_run = normalized_run
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.commands.motion.motion_file = args_cli.motion_file
    if not args_cli.show_debug_vis:
        if hasattr(env_cfg.scene, "contact_forces"):
            env_cfg.scene.contact_forces.debug_vis = False
        env_cfg.commands.motion.debug_vis = False

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    raw_env = env.unwrapped
    if args_cli.video:
        raw_env.metadata["render_fps"] = 50

    log_dir = os.path.dirname(resume_path)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", args_cli.video_folder_name),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(raw_env, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        raw_env = env.unwrapped

    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, adapt_legacy_ppo_cfg(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=raw_env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_motion_policy_as_onnx(
        raw_env,
        get_policy_module(ppo_runner.alg),
        normalizer=getattr(ppo_runner, "obs_normalizer", None),
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(raw_env, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)

    obs, _ = env.reset()
    command = raw_env.command_manager.get_term("motion")
    selected = select_switch_motions(command, split_patterns(args_cli.switch_motions), args_cli.switch_regex)
    print("[INFO] Switch sequence:")
    for index, (label, start, length) in enumerate(selected):
        print(f"  {index}: {label} start={start} length={length}")

    switch_interval = max(int(args_cli.switch_interval), 1)
    max_steps = args_cli.max_steps if args_cli.max_steps is not None else args_cli.video_length
    snap = not args_cli.no_snap_robot_on_switch
    current_switch_index = 0
    apply_switch(raw_env, command, selected, current_switch_index, snap)

    for timestep in range(max_steps):
        if timestep > 0 and timestep % switch_interval == 0:
            current_switch_index += 1
            apply_switch(raw_env, command, selected, current_switch_index, snap)
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        if args_cli.video and timestep + 1 >= args_cli.video_length:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
