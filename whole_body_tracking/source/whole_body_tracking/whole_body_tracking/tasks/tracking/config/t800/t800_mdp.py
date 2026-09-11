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


# Official supine_to_stance entry (traj frame 90). Bridge goal is this FULL state.
T800_FRAME90_JOINTS = [-0.281982421875, -0.0084686279296875, -0.062164306640625, 0.06683349609375, 0.10638427734375, -0.0013532638549804688, -0.26708984375, 0.01029205322265625, 0.05535888671875, 0.039642333984375, 0.11444091796875, 0.0010042190551757812, -0.0020351409912109375, 0.484375, 0.018707275390625, 0.0159149169921875, -0.787109375, 0.00238037109375, 0.480712890625, -0.0216522216796875, -0.007144927978515625, -0.78271484375, -0.0154876708984375, -0.18701171875, 0.0005230903625488281]
T800_FRAME90_BASE_POS = [-0.00026416778564453125, 0.0548095703125, 0.14013671875]
T800_FRAME90_BASE_QUAT_WXYZ = [0.55078125, -0.4443359375, -0.44287109375, -0.55029296875]
T800_FRAME90_TARGET_HEIGHT = T800_FRAME90_BASE_POS[2]



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



def getup_height_upright_gated_max_joint_progress(
    env: "ManagerBasedEnv",
    min_height: float,
    max_tilt_rad: float,
    target_joint_pos: list[float],
    start_error: float,
    target_error: float,
    height_temperature: float,
    tilt_temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense progress on max joint error once the robot is high and roughly upright."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    root_height = root_pos[:, 2] - env_origins[:, 2]
    gravity_z = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(gravity_z)
    height_gate = torch.sigmoid((root_height - min_height) / height_temperature)
    tilt_gate = torch.sigmoid((max_tilt_rad - tilt) / tilt_temperature)
    joint_error = torch.max(torch.abs(getup_target_joint_error(env, target_joint_pos, asset_cfg)), dim=-1).values
    denom = max(float(start_error - target_error), 1.0e-6)
    progress = torch.clamp((start_error - joint_error) / denom, min=0.0, max=1.0)
    return height_gate * tilt_gate * progress


def getup_height_upright_gated_root_low_velocity(
    env: "ManagerBasedEnv",
    min_height: float,
    max_tilt_rad: float,
    velocity_std: float,
    height_temperature: float,
    tilt_temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root low-velocity reward gated by height and uprightness (no joint-vel product)."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    root_height = root_pos[:, 2] - env_origins[:, 2]
    gravity_z = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(gravity_z)
    height_gate = torch.sigmoid((root_height - min_height) / height_temperature)
    tilt_gate = torch.sigmoid((max_tilt_rad - tilt) / tilt_temperature)
    root_vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=-1)
    root_vel_sq += 0.25 * torch.sum(torch.square(asset.data.root_ang_vel_b), dim=-1)
    return height_gate * tilt_gate * torch.exp(-root_vel_sq / velocity_std**2)


def getup_height_upright_gated_joint_low_velocity(
    env: "ManagerBasedEnv",
    min_height: float,
    max_tilt_rad: float,
    joint_velocity_std: float,
    height_temperature: float,
    tilt_temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint low-velocity reward gated by height and uprightness."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    root_height = root_pos[:, 2] - env_origins[:, 2]
    gravity_z = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(gravity_z)
    height_gate = torch.sigmoid((root_height - min_height) / height_temperature)
    tilt_gate = torch.sigmoid((max_tilt_rad - tilt) / tilt_temperature)
    joint_vel_sq = torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)
    return height_gate * tilt_gate * torch.exp(-joint_vel_sq / joint_velocity_std**2)


def getup_low_root_velocity_exp(
    env: "ManagerBasedEnv",
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=-1)
    vel_sq += 0.25 * torch.sum(torch.square(asset.data.root_ang_vel_b), dim=-1)
    return torch.exp(-vel_sq / std**2)



def getup_stand_stable_exp(
    env: "ManagerBasedEnv",
    target_height: float,
    max_tilt_rad: float,
    max_height_error: float,
    root_speed_comfort: float,
    height_temperature: float = 0.05,
    tilt_temperature: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward stable standing without requiring a tight boxing joint match.

    Small root motions for balance are tolerated via root_speed_comfort.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_z = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(gravity_z)
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    height_error = torch.abs(root_pos[:, 2] - env_origins[:, 2] - target_height)
    height_gate = torch.sigmoid((max_height_error - height_error) / height_temperature)
    tilt_gate = torch.sigmoid((max_tilt_rad - tilt) / tilt_temperature)
    root_speed = torch.linalg.norm(asset.data.root_lin_vel_b, dim=-1)
    # 1 at zero speed, ~0.6 around comfort speed, decays beyond.
    speed_term = torch.exp(-torch.square(root_speed) / (root_speed_comfort**2))
    return height_gate * tilt_gate * speed_term



def getup_named_subset_joint_pose_exp(
    env: "ManagerBasedEnv",
    min_height: float,
    target_joint_pos: list[float],
    joint_names: list[str],
    std: float,
    height_temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Height-gated pose match on a named joint subset (e.g. legs+torso only)."""
    from whole_body_tracking.robots.t800_joint_order import T800_POLICY_JOINT_NAMES

    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    root_height = root_pos[:, 2] - env_origins[:, 2]
    height_gate = torch.sigmoid((root_height - min_height) / height_temperature)

    name_to_idx = {name: i for i, name in enumerate(T800_POLICY_JOINT_NAMES)}
    missing = [n for n in joint_names if n not in name_to_idx]
    if missing:
        raise ValueError(f"Unknown joint names for subset pose reward: {missing}")
    target_idx = [name_to_idx[n] for n in joint_names]
    # Resolve articulation joint ids in the requested name order.
    joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    target = _as_pose_tensor(target_joint_pos, joint_pos.device)[target_idx]
    err = target.unsqueeze(0) - joint_pos
    return height_gate * torch.exp(-torch.mean(torch.square(err), dim=-1) / std**2)


def getup_stance_width_exp(
    env: "ManagerBasedEnv",
    min_height: float,
    target_width: float,
    std: float,
    height_temperature: float,
    max_width: float = 1.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"], preserve_order=True
    ),
) -> torch.Tensor:
    """Keep feet near boxing-stance width; soft progress from wide splits + Gaussian near target."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    root_height = root_pos[:, 2] - env_origins[:, 2]
    height_gate = torch.sigmoid((root_height - min_height) / height_temperature)
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    width = torch.linalg.norm(body_pos[:, 0] - body_pos[:, 1], dim=-1)
    # Progress: 0 at max_width, 1 at/below target_width (gives gradient while still split).
    progress = torch.clamp((max_width - width) / max(1.0e-6, max_width - target_width), min=0.0, max=1.0)
    # Prefer not being wider than target; mild penalty if too narrow.
    overshoot = torch.relu(width - target_width)
    undershoot = torch.relu(target_width - width)
    near = torch.exp(-torch.square(overshoot + 0.35 * undershoot) / std**2)
    return height_gate * (0.55 * progress + 0.45 * near)


def getup_episode_late_scale(
    env: "ManagerBasedEnv",
    start_frac: float = 0.45,
) -> torch.Tensor:
    """0 early, 1 in the last portion of the episode ??used to emphasize terminal hold."""
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    return torch.clamp((progress - start_frac) / max(1.0e-6, 1.0 - start_frac), min=0.0, max=1.0)


def getup_stand_baoquan_hold_exp(
    env: "ManagerBasedEnv",
    target_height: float,
    target_joint_pos: list[float],
    leg_joint_names: list[str],
    max_tilt_rad: float,
    max_height_error: float,
    leg_std: float,
    root_speed_comfort: float,
    start_frac: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Late-episode hold: upright + height + leg/torso baoquan stance + soft root speed."""
    from whole_body_tracking.robots.t800_joint_order import T800_POLICY_JOINT_NAMES

    asset: Articulation = env.scene[asset_cfg.name]
    gravity_z = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(gravity_z)
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    height_error = torch.abs(root_pos[:, 2] - env_origins[:, 2] - target_height)
    height_gate = torch.sigmoid((0.18 - height_error) / 0.05)
    tilt_gate = torch.sigmoid((max_tilt_rad - tilt) / 0.08)
    root_speed = torch.linalg.norm(asset.data.root_lin_vel_b, dim=-1)
    speed_term = torch.exp(-torch.square(root_speed) / (root_speed_comfort**2))

    name_to_idx = {name: i for i, name in enumerate(T800_POLICY_JOINT_NAMES)}
    target_idx = [name_to_idx[n] for n in leg_joint_names]
    joint_ids, _ = asset.find_joints(leg_joint_names, preserve_order=True)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    target = _as_pose_tensor(target_joint_pos, joint_pos.device)[target_idx]
    err = target.unsqueeze(0) - joint_pos
    leg_term = torch.exp(-torch.mean(torch.square(err), dim=-1) / leg_std**2)
    late = getup_episode_late_scale(env, start_frac=start_frac)
    return late * height_gate * tilt_gate * speed_term * leg_term



def getup_stance_geometry_exp(
    env: "ManagerBasedEnv",
    min_height: float,
    target_lat: float,
    lat_std: float,
    min_lat: float,
    target_stagger: float,
    stagger_std: float,
    height_temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"], preserve_order=True
    ),
) -> torch.Tensor:
    """Body-frame foot geometry for boxing stance: uncrossed lateral base + forward stagger.

    Uses root yaw frame: +x forward, +y left. Requires left ankle to the left of right ankle
    (lat = y_L - y_R > min_lat) to kill crossed-leg terminals.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    root_height = root_pos[:, 2] - env_origins[:, 2]
    height_gate = torch.sigmoid((root_height - min_height) / height_temperature)

    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    left = body_pos[:, 0] - root_pos
    right = body_pos[:, 1] - root_pos
    left_b = math_utils.quat_rotate_inverse(root_quat, left)
    right_b = math_utils.quat_rotate_inverse(root_quat, right)

    lat = left_b[:, 1] - right_b[:, 1]  # +y left: positive => uncrossed
    stagger = torch.abs(left_b[:, 0] - right_b[:, 0])

    uncrossed = torch.sigmoid((lat - min_lat) / 0.04)
    lat_term = torch.exp(-torch.square(lat - target_lat) / lat_std**2)
    stagger_term = torch.exp(-torch.square(stagger - target_stagger) / stagger_std**2)
    return height_gate * uncrossed * lat_term * stagger_term


def getup_hip_knee_baoquan_exp(
    env: "ManagerBasedEnv",
    min_height: float,
    target_joint_pos: list[float],
    std: float,
    height_temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tight match on hip roll/yaw + knees ??the joints that define uncrossed crouched guard."""
    joint_names = [
        "J01_HIP_ROLL_L",
        "J02_HIP_YAW_L",
        "J03_KNEE_PITCH_L",
        "J07_HIP_ROLL_R",
        "J08_HIP_YAW_R",
        "J09_KNEE_PITCH_R",
    ]
    return getup_named_subset_joint_pose_exp(
        env,
        min_height=min_height,
        target_joint_pos=target_joint_pos,
        joint_names=joint_names,
        std=std,
        height_temperature=height_temperature,
        asset_cfg=asset_cfg,
    )


def getup_knee_flex_floor_exp(
    env: "ManagerBasedEnv",
    min_height: float,
    min_knee_rad: float,
    height_temperature: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Keep knees bent like a guard crouch (avoid locked-straight legs)."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    root_height = root_pos[:, 2] - env_origins[:, 2]
    height_gate = torch.sigmoid((root_height - min_height) / height_temperature)
    joint_ids, _ = asset.find_joints(["J03_KNEE_PITCH_L", "J09_KNEE_PITCH_R"], preserve_order=True)
    knees = asset.data.joint_pos[:, joint_ids]
    # Measured baoquan knees are ~+0.76/+0.82; require both above min_knee_rad.
    left_ok = torch.sigmoid((knees[:, 0] - min_knee_rad) / 0.08)
    right_ok = torch.sigmoid((knees[:, 1] - min_knee_rad) / 0.08)
    return height_gate * left_ok * right_ok


def getup_terminal_baoquan_hold_exp(
    env: "ManagerBasedEnv",
    target_height: float,
    target_joint_pos: list[float],
    leg_joint_names: list[str],
    max_tilt_rad: float,
    max_height_error: float,
    leg_std: float,
    root_speed_comfort: float,
    min_lat: float,
    target_lat: float,
    start_frac: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"], preserve_order=True
    ),
) -> torch.Tensor:
    """Late-episode hold: upright + height + leg pose + uncrossed stance width."""
    from whole_body_tracking.robots.t800_joint_order import T800_POLICY_JOINT_NAMES

    asset: Articulation = env.scene[asset_cfg.name]
    gravity_z = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(gravity_z)
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w
    env_origins = env.scene.env_origins.to(root_pos.device)
    height_error = torch.abs(root_pos[:, 2] - env_origins[:, 2] - target_height)
    height_gate = torch.sigmoid((0.18 - height_error) / 0.05)
    tilt_gate = torch.sigmoid((max_tilt_rad - tilt) / 0.08)
    root_speed = torch.linalg.norm(asset.data.root_lin_vel_b, dim=-1)
    speed_term = torch.exp(-torch.square(root_speed) / (root_speed_comfort**2))

    name_to_idx = {name: i for i, name in enumerate(T800_POLICY_JOINT_NAMES)}
    target_idx = [name_to_idx[n] for n in leg_joint_names]
    joint_ids, _ = asset.find_joints(leg_joint_names, preserve_order=True)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    target = _as_pose_tensor(target_joint_pos, joint_pos.device)[target_idx]
    err = target.unsqueeze(0) - joint_pos
    leg_term = torch.exp(-torch.mean(torch.square(err), dim=-1) / leg_std**2)

    body_pos = asset.data.body_pos_w[:, foot_cfg.body_ids, :]
    left_b = math_utils.quat_rotate_inverse(root_quat, body_pos[:, 0] - root_pos)
    right_b = math_utils.quat_rotate_inverse(root_quat, body_pos[:, 1] - root_pos)
    lat = left_b[:, 1] - right_b[:, 1]
    stance_term = torch.sigmoid((lat - min_lat) / 0.04) * torch.exp(
        -torch.square(lat - target_lat) / (0.12**2)
    )

    late = getup_episode_late_scale(env, start_frac=start_frac)
    return late * height_gate * tilt_gate * speed_term * leg_term * stance_term


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


def reset_t800_bridge_pose(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    start_pose: str,
    root_height: float,
    pose_noise: dict[str, tuple[float, float]],
    joint_position_noise: tuple[float, float],
    velocity_noise: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset near pd_stand_x / pd_stand_y floor poses for supine-entry bridging.

    start_pose:
      - "pose_x" / "prone": competition X prep joints + prone-like base
      - "pose_y" / "supine": competition Y prep joints + supine-like base
      - "mixed": half X, half Y
    """
    orientation = {
        "pose_x": "prone",
        "pose_y": "supine",
        "prone": "prone",
        "supine": "supine",
        "mixed": "mixed",
    }.get(start_pose, start_pose)
    return reset_t800_getup_pose(
        env,
        env_ids,
        orientation=orientation,
        root_height=root_height,
        pose_noise=pose_noise,
        joint_position_noise=joint_position_noise,
        velocity_noise=velocity_noise,
        asset_cfg=asset_cfg,
    )


def bridge_root_pos_error(
    env: "ManagerBasedEnv",
    target_base_pos: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """World-frame root position error vs frame90 base_pos (env-local)."""
    asset: Articulation = env.scene[asset_cfg.name]
    target = _as_pose_tensor(target_base_pos, asset.device).view(1, 3)
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    return root_pos - target


def bridge_quat_dot(
    env: "ManagerBasedEnv",
    target_base_quat_wxyz: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    target = _as_pose_tensor(target_base_quat_wxyz, asset.device).view(1, 4)
    root_quat = asset.data.root_quat_w
    return torch.abs(torch.sum(root_quat * target, dim=-1))


def bridge_quat_error_obs(
    env: "ManagerBasedEnv",
    target_base_quat_wxyz: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation: 1 - |q?q_tgt| as a 1-D feature (plus reused with height)."""
    return (1.0 - bridge_quat_dot(env, target_base_quat_wxyz, asset_cfg)).unsqueeze(-1)


def bridge_ori_exp(
    env: "ManagerBasedEnv",
    target_base_quat_wxyz: list[float],
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    err = 1.0 - bridge_quat_dot(env, target_base_quat_wxyz, asset_cfg)
    return torch.exp(-err / (std * std))


def bridge_base_pos_exp(
    env: "ManagerBasedEnv",
    target_base_pos: list[float],
    std: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    err = torch.linalg.norm(bridge_root_pos_error(env, target_base_pos, asset_cfg), dim=-1)
    return torch.exp(-err / (std * std))


def bridge_success_bonus(
    env: "ManagerBasedEnv",
    target_joint_pos: list[float],
    target_base_pos: list[float],
    target_base_quat_wxyz: list[float],
    max_joint_error: float = 0.15,
    max_pos_error: float = 0.08,
    min_quat_dot: float = 0.95,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    joint_err = torch.max(torch.abs(getup_target_joint_error(env, target_joint_pos, asset_cfg)), dim=-1).values
    pos_err = torch.linalg.norm(bridge_root_pos_error(env, target_base_pos, asset_cfg), dim=-1)
    quat_dot = bridge_quat_dot(env, target_base_quat_wxyz, asset_cfg)
    ok = (joint_err < max_joint_error) & (pos_err < max_pos_error) & (quat_dot > min_quat_dot)
    return ok.to(dtype=torch.float32)


