"""Command curriculum for the fixed-guard staged locomotion plan.

Stage index is owned by the external watchdog JSON so DDP ranks stay aligned.
This term only applies the corresponding velocity-command ranges.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

CURRICULUM_STAGE_NAMES = (
    "standing_balance",
    "forward_warmup",
    "backward_expansion",
    "lateral_expansion",
    "yaw_mixed",
    "light_terrain",
    "speed_bump",
)

_DEFAULT_STATE = Path(
    "/mnt/data/yangky/test/humanoid_robot_combat/whole_body_tracking/"
    "results/t800_fixed_guard72_20260910/pipeline/state.json"
)


def enforce_minimum_moving_speed(
    command: torch.Tensor,
    standing: torch.Tensor,
    minimum_speed: float,
) -> torch.Tensor:
    """Project non-standing commands out of the near-zero reward-hacking region."""
    if minimum_speed <= 0.0:
        raise ValueError("minimum_speed must be positive")
    moving = ~standing
    too_slow = moving & (torch.linalg.vector_norm(command, dim=1) < minimum_speed)
    command[too_slow, 0] = torch.where(
        command[too_slow, 0] < 0.0,
        -minimum_speed,
        minimum_speed,
    )
    command[standing] = 0.0
    return command


def _load_stage_state() -> dict:
    path = Path(os.environ.get("FG72_STATE_JSON", str(_DEFAULT_STATE)))
    if not path.is_file():
        return {"stage": 0, "command_scale": 1.0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"stage": 0, "command_scale": 1.0}
    stage = int(payload.get("stage", 0))
    scale = float(payload.get("command_scale", 1.0))
    stage = max(0, min(stage, len(CURRICULUM_STAGE_NAMES) - 1))
    scale = min(max(scale, 0.35), 1.0)
    mode = str(payload.get("command_mode") or os.environ.get("FG72_CMD_MODE", "omni"))
    return {"stage": stage, "command_scale": scale, "command_mode": mode}


def _apply_stage(ranges, command_term, stage: int, scale: float) -> None:
    s = scale
    if stage <= 0:
        ranges.lin_vel_x = (0.0, 0.0)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)
        command_term.cfg.rel_standing_envs = 1.0
        return
    if stage == 1:
        ranges.lin_vel_x = (0.15 * s, 0.45 * s)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)
        command_term.cfg.rel_standing_envs = 0.25
        return
    if stage == 2:
        ranges.lin_vel_x = (-0.35 * s, 0.70 * s)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)
        command_term.cfg.rel_standing_envs = 0.12
        return
    if stage == 3:
        ranges.lin_vel_x = (-0.70 * s, 0.80 * s)
        ranges.lin_vel_y = (-0.50 * s, 0.50 * s)
        ranges.ang_vel_z = (0.0, 0.0)
        command_term.cfg.rel_standing_envs = 0.08
        return
    # stage 4+ : default omni stick. back_yaw drops sidestep, the motion that
    # crossed the feet in play, and keeps retreat + turn which already work.
    mode = str(_load_stage_state().get("command_mode", "omni"))
    if mode == "back_yaw":
        ranges.lin_vel_x = (-0.70 * s, 0.12 * s)
        ranges.lin_vel_y = (-0.05 * s, 0.05 * s)
        ranges.ang_vel_z = (-0.85 * s, 0.85 * s)
        command_term.cfg.rel_standing_envs = 0.08
        return
    ranges.lin_vel_x = (-0.80 * s, 0.80 * s)
    ranges.lin_vel_y = (-0.60 * s, 0.60 * s)
    ranges.ang_vel_z = (-0.80 * s, 0.80 * s)
    command_term.cfg.rel_standing_envs = 0.05


def velocity_command_curriculum(
    env,
    env_ids,
    warmup_steps: int = 200_000,
    standing_fraction: float = 0.20,
    forward_only_fraction: float = 0.40,
    backward_fraction: float = 0.60,
    lateral_fraction: float = 0.80,
    initial_stage: int = 0,
):
    """Apply watchdog-selected stage ranges. Unused time-fraction args kept for cfg compatibility."""
    del warmup_steps, standing_fraction, forward_only_fraction, backward_fraction, lateral_fraction
    state = _load_stage_state()
    stage = max(int(initial_stage), state["stage"])
    stage = min(stage, len(CURRICULUM_STAGE_NAMES) - 1)
    scale = state["command_scale"]
    command_term = env.command_manager.get_term("base_velocity")
    _apply_stage(command_term.cfg.ranges, command_term, stage, scale)
    env.fixed_guard_curriculum_stage = CURRICULUM_STAGE_NAMES[stage]
    return {
        "progress": float(stage) / float(len(CURRICULUM_STAGE_NAMES) - 1),
        "direction_progress": float(scale),
        "stage_index": float(stage),
        "command_scale": float(scale),
    }
