from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg


def upper_body_guard_error(env, asset_cfg: SceneEntityCfg, target_joint_pos: list[float]) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    target = torch.as_tensor(target_joint_pos, device=asset.device, dtype=asset.data.joint_pos.dtype).unsqueeze(0)
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - target), dim=-1)


def upper_body_guard_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)


def base_height_l2_clipped(env, target_height: float, max_error: float = 0.25, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.clamp(torch.square(asset.data.root_pos_w[:, 2] - target_height), max=max_error)

def unstable_root(env, target_height: float, max_height_error: float = 0.6, max_linear_speed: float = 8.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    bad_height = torch.abs(asset.data.root_pos_w[:, 2] - target_height) > max_height_error
    bad_speed = torch.linalg.norm(asset.data.root_lin_vel_w, dim=-1) > max_linear_speed
    return bad_height | bad_speed
