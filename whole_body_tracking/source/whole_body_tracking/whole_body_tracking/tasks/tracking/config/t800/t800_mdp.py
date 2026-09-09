from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

from whole_body_tracking.robots.t800_joint_order import T800_DFS_JOINT_NAMES
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


T800_MOTION_BODY_NAMES = [
    "LINK_BASE",
    "LINK_HIP_PITCH_L",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_YAW_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_L_TOE",
    "LINK_ANKLE_ROLL_L_HEEL",
    "LINK_HIP_PITCH_R",
    "LINK_HIP_ROLL_R",
    "LINK_HIP_YAW_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_ANKLE_ROLL_R_TOE",
    "LINK_ANKLE_ROLL_R_HEEL",
    "LINK_TORSO_YAW",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    "LINK_WRIST_PITCH_L",
    "LINK_WRIST_ROLL_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_SHOULDER_ROLL_R",
    "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
    "LINK_WRIST_PITCH_R",
    "LINK_WRIST_ROLL_R",
    "LINK_HEAD_PITCH",
    "LINK_HEAD_YAW",
]

T800_PD_STAND_X = [
    -1.7,
    1.13,
    2.85,
    0.582,
    0.267,
    -0.35,
    -0.153,
    0.631,
    -1.93,
    0.499,
    -0.4,
    -0.175,
    -0.216,
    0.821,
    2.04,
    -1.74,
    -0.258,
    -0.803,
    2.59,
    -1.88,
    -0.944,
    0.262,
    0.423,
    0.0866,
    1.22,
]

T800_PD_STAND_Y = [
    0.392,
    0.594,
    -0.891,
    1.49,
    0.481,
    0.175,
    -1.79,
    -1.05,
    0.412,
    2.36,
    0.482,
    0.35,
    0.363,
    0.183,
    0.129,
    1.2,
    -0.362,
    0.207,
    1.46,
    -1.36,
    -0.192,
    -0.669,
    -0.2,
    -0.478,
    0.179,
]

# Placeholder target for direct-RL bring-up. Replace this list with measured
# vendor boxing-idle telemetry before using any direct get-up model on hardware.
T800_APPROX_BOXING_READY = [
    -0.28,
    0.08,
    0.02,
    0.58,
    -0.30,
    -0.04,
    -0.28,
    -0.08,
    -0.02,
    0.58,
    -0.30,
    0.04,
    0.0,
    0.35,
    0.85,
    -0.25,
    -1.05,
    -0.35,
    0.35,
    -0.85,
    0.25,
    -1.05,
    0.35,
    0.0,
    0.0,
]


def _as_pose_tensor(values: list[float], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def _resolve_getup_initial_pose(orientation: str) -> list[float]:
    if orientation == "prone":
        return T800_PD_STAND_X
    if orientation == "supine":
        return T800_PD_STAND_Y
    raise ValueError(f"Unsupported T800 get-up orientation: {orientation!r}")


def _get_joint_pos(asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def reset_t800_getup_pose(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    orientation: str,
    root_height: float,
    pose_noise: dict[str, tuple[float, float]],
    joint_position_noise: tuple[float, float],
    velocity_noise: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the T800 near the official prone/supine PD preparation pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    range_list = [pose_noise.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, dtype=torch.float32, device=asset.device)
    rand = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=asset.device)

    root_pos = env.scene.env_origins[env_ids].clone()
    root_pos[:, 0:3] += rand[:, 0:3]
    root_pos[:, 2] += root_height

    if orientation == "prone":
        base_roll = torch.zeros(num_envs, device=asset.device)
        base_pitch = torch.full((num_envs,), torch.pi, device=asset.device)
    elif orientation == "supine":
        base_roll = torch.full((num_envs,), torch.pi, device=asset.device)
        base_pitch = torch.zeros(num_envs, device=asset.device)
    else:
        half = num_envs // 2
        base_roll = torch.zeros(num_envs, device=asset.device)
        base_pitch = torch.full((num_envs,), torch.pi, device=asset.device)
        base_roll[half:] = torch.pi
        base_pitch[half:] = 0.0

    base_yaw = torch.zeros(num_envs, device=asset.device)
    base_quat = math_utils.quat_from_euler_xyz(base_roll, base_pitch, base_yaw)
    delta_quat = math_utils.quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5])
    root_quat = math_utils.quat_mul(delta_quat, base_quat)
    root_vel = math_utils.sample_uniform(*velocity_noise, (num_envs, 6), device=asset.device)

    if orientation == "mixed":
        initial = torch.empty((num_envs, len(T800_PD_STAND_X)), dtype=torch.float32, device=asset.device)
        initial[:half] = _as_pose_tensor(T800_PD_STAND_X, asset.device)
        initial[half:] = _as_pose_tensor(T800_PD_STAND_Y, asset.device)
    else:
        initial = _as_pose_tensor(_resolve_getup_initial_pose(orientation), asset.device).repeat(num_envs, 1)
    joint_pos = initial + math_utils.sample_uniform(
        *joint_position_noise, initial.shape, device=asset.device
    )
    joint_limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos = torch.clamp(joint_pos, joint_limits[:, :, 0], joint_limits[:, :, 1])
    joint_vel = math_utils.sample_uniform(*velocity_noise, joint_pos.shape, device=asset.device)

    asset.write_root_state_to_sim(torch.cat([root_pos, root_quat, root_vel], dim=-1), env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def getup_target_joint_error(
    env: "ManagerBasedEnv",
    target_joint_pos: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = _get_joint_pos(asset, asset_cfg)
    target = _as_pose_tensor(target_joint_pos, joint_pos.device)
    return target.unsqueeze(0) - joint_pos


def getup_root_height_error(
    env: "ManagerBasedEnv",
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    return (root_pos[:, 2] - env_origins[:, 2] - target_height).unsqueeze(-1)


def getup_target_joint_pose_exp(
    env: "ManagerBasedEnv",
    target_joint_pos: list[float],
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    err = getup_target_joint_error(env, target_joint_pos, asset_cfg)
    return torch.exp(-torch.mean(torch.square(err), dim=-1) / std**2)


def getup_root_height_exp(
    env: "ManagerBasedEnv",
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    err = getup_root_height_error(env, target_height, asset_cfg).squeeze(-1)
    return torch.exp(-torch.square(err) / std**2)


def getup_upright_exp(
    env: "ManagerBasedEnv",
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_z = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(gravity_z)
    return torch.exp(-torch.square(tilt) / std**2)


def getup_upright_linear(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_z = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    return torch.clamp(0.5 * (gravity_z + 1.0), min=0.0, max=1.0)


def getup_root_height_linear(
    env: "ManagerBasedEnv",
    target_height: float,
    max_error: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    err = torch.abs(getup_root_height_error(env, target_height, asset_cfg).squeeze(-1))
    return torch.clamp(1.0 - err / max_error, min=0.0, max=1.0)


def getup_root_height_stage_reward(
    env: "ManagerBasedEnv",
    thresholds: list[float],
    temperature: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    root_height = root_pos[:, 2] - env_origins[:, 2]
    levels = torch.tensor(thresholds, dtype=torch.float32, device=root_height.device)
    rewards = torch.sigmoid((root_height.unsqueeze(-1) - levels.unsqueeze(0)) / temperature)
    return torch.mean(rewards, dim=-1)


def getup_stability_exp(
    env: "ManagerBasedEnv",
    min_height: float,
    velocity_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    root_height = root_pos[:, 2] - env_origins[:, 2]
    height_gate = torch.sigmoid((root_height - min_height) / 0.05)
    vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=-1)
    vel_sq += 0.25 * torch.sum(torch.square(asset.data.root_ang_vel_b), dim=-1)
    return height_gate * torch.exp(-vel_sq / velocity_std**2)


def getup_guard_stability_exp(
    env: "ManagerBasedEnv",
    target_height: float,
    target_joint_pos: list[float],
    max_tilt_rad: float,
    max_height_error: float,
    max_joint_error: float,
    velocity_std: float,
    joint_velocity_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_z = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(gravity_z)
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    height_error = torch.abs(root_pos[:, 2] - env_origins[:, 2] - target_height)
    joint_error = torch.max(torch.abs(getup_target_joint_error(env, target_joint_pos, asset_cfg)), dim=-1).values

    height_gate = torch.sigmoid((max_height_error - height_error) / 0.04)
    tilt_gate = torch.sigmoid((max_tilt_rad - tilt) / 0.06)
    joint_gate = torch.sigmoid((max_joint_error - joint_error) / 0.08)
    root_vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=-1)
    root_vel_sq += 0.25 * torch.sum(torch.square(asset.data.root_ang_vel_b), dim=-1)
    joint_vel_sq = torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)
    stability = torch.exp(-root_vel_sq / velocity_std**2) * torch.exp(-joint_vel_sq / joint_velocity_std**2)
    return height_gate * tilt_gate * joint_gate * stability


def getup_height_gated_upright_exp(
    env: "ManagerBasedEnv",
    min_height: float,
    std: float,
    height_temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    root_height = root_pos[:, 2] - env_origins[:, 2]
    height_gate = torch.sigmoid((root_height - min_height) / height_temperature)
    gravity_z = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(gravity_z)
    return height_gate * torch.exp(-torch.square(tilt) / std**2)


def getup_height_gated_joint_pose_exp(
    env: "ManagerBasedEnv",
    min_height: float,
    target_joint_pos: list[float],
    std: float,
    height_temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    root_height = root_pos[:, 2] - env_origins[:, 2]
    height_gate = torch.sigmoid((root_height - min_height) / height_temperature)
    err = getup_target_joint_error(env, target_joint_pos, asset_cfg)
    return height_gate * torch.exp(-torch.mean(torch.square(err), dim=-1) / std**2)


def getup_height_gated_low_velocity_exp(
    env: "ManagerBasedEnv",
    min_height: float,
    velocity_std: float,
    joint_velocity_std: float,
    height_temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    root_height = root_pos[:, 2] - env_origins[:, 2]
    height_gate = torch.sigmoid((root_height - min_height) / height_temperature)
    root_vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=-1)
    root_vel_sq += 0.25 * torch.sum(torch.square(asset.data.root_ang_vel_b), dim=-1)
    joint_vel_sq = torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)
    return height_gate * torch.exp(-root_vel_sq / velocity_std**2) * torch.exp(
        -joint_vel_sq / joint_velocity_std**2
    )


def joint_soft_limit_margin_violation(
    env: "ManagerBasedEnv",
    margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = _get_joint_pos(asset, asset_cfg)
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    lower = limits[..., 0]
    upper = limits[..., 1]
    center = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    safe_half_range = half_range * (1.0 - margin)
    normalized = torch.abs(joint_pos - center) / torch.clamp(safe_half_range, min=1.0e-6)
    return torch.mean(torch.square(torch.relu(normalized - 1.0)), dim=-1)


def joint_velocity_limit_violation(
    env: "ManagerBasedEnv",
    max_fraction: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids]
    normalized = torch.abs(joint_vel) / torch.clamp(limits * max_fraction, min=1.0e-6)
    return torch.mean(torch.square(torch.relu(normalized - 1.0)), dim=-1)


def joint_torque_limit_violation(
    env: "ManagerBasedEnv",
    max_fraction: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    normalized = torch.abs(torque) / torch.clamp(limits * max_fraction, min=1.0e-6)
    return torch.mean(torch.square(torch.relu(normalized - 1.0)), dim=-1)


def getup_low_root_velocity_exp(
    env: "ManagerBasedEnv",
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=-1)
    vel_sq += 0.25 * torch.sum(torch.square(asset.data.root_ang_vel_b), dim=-1)
    return torch.exp(-vel_sq / std**2)


def getup_success_bonus(
    env: "ManagerBasedEnv",
    target_height: float,
    target_joint_pos: list[float],
    max_tilt_rad: float,
    max_height_error: float,
    max_joint_error: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_z = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(gravity_z)
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    height_error = torch.abs(root_pos[:, 2] - env_origins[:, 2] - target_height)
    joint_error = torch.max(torch.abs(getup_target_joint_error(env, target_joint_pos, asset_cfg)), dim=-1).values
    success = (tilt < max_tilt_rad) & (height_error < max_height_error) & (joint_error < max_joint_error)
    return success.float()


def getup_root_xy_out_of_bounds(
    env: "ManagerBasedEnv",
    max_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    xy = root_pos[:, :2] - env_origins[:, :2]
    return torch.linalg.norm(xy, dim=-1) > max_distance


def getup_head_contact(
    env: "ManagerBasedEnv",
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    force = torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    return torch.any(torch.max(force, dim=1).values > threshold, dim=1)


class ResidualRefJointPositionAction(JointAction):
    """Apply reference joint targets plus policy residuals."""

    cfg: "ResidualRefJointPositionActionCfg"

    def __init__(self, cfg: "ResidualRefJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._command_name = cfg.command_name
        # Keep an indexable offset tensor so shared startup randomization code does not fail.
        self._offset = torch.zeros((env.num_envs, self.action_dim), dtype=torch.float32, device=env.device)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # Residual action should not inherit the articulation default pose as an offset.
        self._processed_actions = self._raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        command: MotionCommand = self._env.command_manager.get_term(self._command_name)
        target_joint_pos = command.joint_pos + self.processed_actions
        self._asset.set_joint_position_target(target_joint_pos, joint_ids=self._joint_ids)


@configclass
class ResidualRefJointPositionActionCfg(JointActionCfg):
    class_type: type = ResidualRefJointPositionAction
    command_name: str = "motion"
