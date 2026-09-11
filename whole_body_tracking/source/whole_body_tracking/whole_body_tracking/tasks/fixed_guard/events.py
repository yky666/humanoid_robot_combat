"""Reset helpers for the fixed T800 guard pose."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot", joint_names=[".*"])


def reset_guard_joints(
    env,
    env_ids: torch.Tensor,
    guard_position: list[float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids == slice(None):
        raise ValueError("reset_guard_joints requires explicit guard joint names")
    joint_ids = torch.as_tensor(
        asset_cfg.joint_ids, dtype=torch.long, device=asset.device
    )
    target = torch.as_tensor(
        guard_position, dtype=asset.data.joint_pos.dtype, device=asset.device
    )
    if target.numel() != joint_ids.numel() or not torch.isfinite(target).all():
        raise ValueError("guard reset target shape or finiteness is invalid")
    target = target.unsqueeze(0).expand(len(env_ids), -1).clone()
    limits = asset.data.soft_joint_pos_limits[env_ids[:, None], joint_ids]
    target.clamp_(limits[..., 0], limits[..., 1])
    velocity = torch.zeros_like(target)
    asset.write_joint_state_to_sim(
        target, velocity, joint_ids=joint_ids, env_ids=env_ids
    )
