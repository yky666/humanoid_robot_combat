"""Guard-specific reward terms."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from . import gait


def guard_pose_l2(env, target: list[float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    target_tensor = torch.as_tensor(
        target, device=asset.device, dtype=asset.data.joint_pos.dtype
    )
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - target_tensor
    return torch.sum(torch.square(error), dim=1)


def guard_velocity_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def gait_phase_observation(env, cycle_s: float = 0.85) -> torch.Tensor:
    """Expose the same bilateral gait clock used by the reward terms."""
    return gait.phase_features(env.episode_length_buf, env.step_dt, cycle_s)


def feet_positions_b(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return ordered left/right foot positions expressed in the base frame."""
    asset = env.scene[asset_cfg.name]
    foot_delta_w = (
        asset.data.body_pos_w[:, asset_cfg.body_ids] - asset.data.root_pos_w[:, None, :]
    )
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(
        -1, foot_delta_w.shape[1], -1
    )
    foot_pos_b = quat_apply_inverse(
        root_quat_w.reshape(-1, 4), foot_delta_w.reshape(-1, 3)
    )
    return foot_pos_b.reshape(foot_delta_w.shape[0], -1)


def _foot_contact_and_load(
    sensor, body_ids, threshold: float = 1.0, *, allow_air_time: bool = True
):
    """Read contact state and optional force magnitude without fabricating Newtons.

    Isaac Lab's contact sensor exposes ``net_forces_w`` in simulation.  A real
    T800 deployment does not expose that tensor through the SDK, so the reward
    code also accepts the sensor's binary contact/air-time fields and returns
    ``None`` for the unavailable force magnitude.
    """
    data = sensor.data
    force = getattr(data, "net_forces_w", None)
    if force is not None:
        selected_force = force[:, body_ids]
        load = torch.linalg.vector_norm(selected_force, dim=-1)
        return load > threshold, load
    contact_time = getattr(data, "current_contact_time", None)
    if contact_time is not None:
        return contact_time[:, body_ids] > 0.0, None
    if allow_air_time:
        air_time = getattr(data, "current_air_time", None)
        if air_time is not None:
            return air_time[:, body_ids] <= 0.0, None
    raise AttributeError(
        "foot contact sensor needs net_forces_w, current_contact_time, or current_air_time"
    )


def _averaged_foot_load(sensor, body_ids):
    """Average simulated force magnitudes over the available history window."""
    data = sensor.data
    force = getattr(data, "net_forces_w", None)
    if force is None:
        return None
    history = getattr(data, "net_forces_w_history", None)
    if history is None:
        return torch.linalg.vector_norm(force[:, body_ids], dim=-1)
    selected_force = history[:, :, body_ids]
    return torch.linalg.vector_norm(selected_force, dim=-1).mean(dim=1)


def feet_slide_safe(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize horizontal foot motion while a foot is in contact.

    This mirrors IsaacLab's ``feet_slide`` term but uses the same binary
    contact/air-time fallback as the guard-specific gait rewards when a
    synthetic or deployment-shaped sensor has no ``net_forces_w`` field.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    contact, _ = _foot_contact_and_load(sensor, sensor_cfg.body_ids, threshold)
    foot_velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return (torch.linalg.vector_norm(foot_velocity, dim=-1) * contact).sum(dim=1)


def illegal_contact_safe(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate on contact without requiring force magnitudes.

    Air-time is intentionally not treated as contact for this term: a torso or
    base body can have zero air-time while standing, which would otherwise
    trigger termination in a no-force sensor path.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    try:
        contact, _ = _foot_contact_and_load(
            sensor, sensor_cfg.body_ids, threshold, allow_air_time=False
        )
    except AttributeError:
        data = sensor.data
        num_envs = getattr(env, "num_envs", None)
        if num_envs is None:
            for field in ("net_forces_w", "current_contact_time", "current_air_time"):
                value = getattr(data, field, None)
                if value is not None:
                    num_envs = int(value.shape[0])
                    break
        if num_envs is None:
            raise
        device = getattr(env, "device", None)
        if device is None:
            for field in ("net_forces_w", "current_contact_time", "current_air_time"):
                value = getattr(data, field, None)
                if value is not None:
                    device = value.device
                    break
        contact = torch.zeros(
            (num_envs, len(sensor_cfg.body_ids)), dtype=torch.bool, device=device
        )
    return contact.any(dim=1)


def _phase_masks(env, cycle_s: float, air_ratio: float, delta_t: float):
    phases = gait.foot_phases(
        env.episode_length_buf,
        env.step_dt,
        cycle_s,
    )
    return phases, *gait.gait_clock(phases, air_ratio, delta_t)


def _gait_force_compliance(
    env,
    sensor_cfg: SceneEntityCfg,
    cycle_s: float,
    air_ratio: float,
    force_scale: float,
    swing_force_std: float,
    stance_force_std: float,
    delta_t: float = 0.02,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    contact, _ = _foot_contact_and_load(sensor, sensor_cfg.body_ids)
    load = _averaged_foot_load(sensor, sensor_cfg.body_ids)
    _, swing_mask, stance_mask = _phase_masks(env, cycle_s, air_ratio, delta_t)
    if load is None:
        swing = gait.smooth_binary_swing_score(contact, swing_mask)
        support = gait.smooth_binary_stance_score(contact, stance_mask)
    else:
        normalized_force = gait.normalized_foot_force(load, force_scale)
        swing = gait.smooth_swing_force_score(
            normalized_force, swing_mask, swing_force_std
        )
        support = gait.smooth_stance_support_force_score(
            normalized_force, stance_mask, stance_force_std
        )
    return swing * support


def track_lin_vel_xy_gait_exp(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    std: float = 0.5,
    gate_floor: float = 0.2,
    force_scale: float = 400.0,
    swing_force_std: float = 0.08,
    stance_force_std: float = 0.15,
    cycle_s: float = 0.85,
    air_ratio: float = 0.38,
    delta_t: float = 0.02,
) -> torch.Tensor:
    """Track XY velocity, gating moving credit by force-based gait compliance."""
    command = env.command_manager.get_command(command_name)
    asset = env.scene["robot"]
    error = torch.sum(
        torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1
    )
    tracking = torch.exp(-error / (std * std))
    compliance = _gait_force_compliance(
        env,
        sensor_cfg,
        cycle_s,
        air_ratio,
        force_scale,
        swing_force_std,
        stance_force_std,
        delta_t,
    )
    return gait.gate_tracking_reward(
        tracking, compliance, gait.command_is_active(command), gate_floor
    )


def track_ang_vel_z_gait_exp(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    std: float = 0.5,
    gate_floor: float = 0.2,
    force_scale: float = 400.0,
    swing_force_std: float = 0.08,
    stance_force_std: float = 0.15,
    cycle_s: float = 0.85,
    air_ratio: float = 0.38,
    delta_t: float = 0.02,
) -> torch.Tensor:
    """Track yaw velocity, gating moving credit by force-based gait compliance."""
    command = env.command_manager.get_command(command_name)
    asset = env.scene["robot"]
    error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    tracking = torch.exp(-error / (std * std))
    compliance = _gait_force_compliance(
        env,
        sensor_cfg,
        cycle_s,
        air_ratio,
        force_scale,
        swing_force_std,
        stance_force_std,
        delta_t,
    )
    return gait.gate_tracking_reward(
        tracking, compliance, gait.command_is_active(command), gate_floor
    )


def command_lin_vel_error_l2(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Return direct XY velocity error so large tracking misses stay visible to PPO."""
    command = env.command_manager.get_command(command_name)
    asset = env.scene["robot"]
    return torch.sum(
        torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1
    )


def command_ang_vel_z_error_l2(
    env, command_name: str = "base_velocity"
) -> torch.Tensor:
    """Return direct yaw-rate error, including spontaneous rotation at zero command."""
    command = env.command_manager.get_command(command_name)
    asset = env.scene["robot"]
    return torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])


def gait_swing_force(
    env,
    sensor_cfg: SceneEntityCfg,
    cycle_s: float = 0.85,
    air_ratio: float = 0.38,
    force_scale: float = 400.0,
    force_std: float = 0.08,
    command_name: str = "base_velocity",
    delta_t: float = 0.02,
) -> torch.Tensor:
    """Reward unloading the clock-selected swing foot.

    Force magnitude is preferred in simulation; binary contact is used when a
    sensor implementation does not provide a force tensor.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    contact, _ = _foot_contact_and_load(sensor, sensor_cfg.body_ids)
    load = _averaged_foot_load(sensor, sensor_cfg.body_ids)
    _, swing_mask, _ = _phase_masks(env, cycle_s, air_ratio, delta_t)
    if load is None:
        score = gait.smooth_binary_swing_score(contact, swing_mask)
    else:
        score = gait.smooth_swing_force_score(
            gait.normalized_foot_force(load, force_scale), swing_mask, force_std
        )
    command = env.command_manager.get_command(command_name)
    return score * gait.command_is_active(command)


def gait_stance_speed(
    env,
    asset_cfg: SceneEntityCfg,
    cycle_s: float = 0.85,
    air_ratio: float = 0.38,
    speed_std: float = 0.15,
    command_name: str = "base_velocity",
    delta_t: float = 0.02,
) -> torch.Tensor:
    """Reward a stationary clock-selected stance foot."""
    asset = env.scene[asset_cfg.name]
    foot_speed = torch.linalg.vector_norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1
    )
    _, _, stance_mask = _phase_masks(env, cycle_s, air_ratio, delta_t)
    score = gait.smooth_stance_velocity_score(foot_speed, stance_mask, speed_std)
    command = env.command_manager.get_command(command_name)
    return score * gait.command_is_active(command)


def gait_stance_support_force(
    env,
    sensor_cfg: SceneEntityCfg,
    cycle_s: float = 0.85,
    air_ratio: float = 0.38,
    force_scale: float = 400.0,
    force_std: float = 0.15,
    command_name: str = "base_velocity",
    delta_t: float = 0.02,
) -> torch.Tensor:
    """Reward loading the clock-selected stance foot."""
    sensor = env.scene.sensors[sensor_cfg.name]
    contact, _ = _foot_contact_and_load(sensor, sensor_cfg.body_ids)
    load = _averaged_foot_load(sensor, sensor_cfg.body_ids)
    _, _, stance_mask = _phase_masks(env, cycle_s, air_ratio, delta_t)
    if load is None:
        score = gait.smooth_binary_stance_score(contact, stance_mask)
    else:
        score = gait.smooth_stance_support_force_score(
            gait.normalized_foot_force(load, force_scale), stance_mask, force_std
        )
    command = env.command_manager.get_command(command_name)
    return score * gait.command_is_active(command)


def gait_contact_phase(
    env,
    sensor_cfg: SceneEntityCfg,
    cycle_s: float = 0.85,
    air_ratio: float = 0.38,
    command_name: str = "base_velocity",
    delta_t: float = 0.02,
) -> torch.Tensor:
    """Reward only an exact, command-gated foot-contact pattern."""
    sensor = env.scene.sensors[sensor_cfg.name]
    contact, _ = _foot_contact_and_load(sensor, sensor_cfg.body_ids)
    phases, _, _ = _phase_masks(env, cycle_s, air_ratio, delta_t)
    expected_contact = phases >= air_ratio
    score = gait.exact_contact_pattern_score(contact, expected_contact)
    command = env.command_manager.get_command(command_name)
    active = gait.command_is_active(command)
    return score * active


def gait_unexpected_double_support(
    env,
    sensor_cfg: SceneEntityCfg,
    cycle_s: float = 0.85,
    air_ratio: float = 0.38,
    command_name: str = "base_velocity",
    delta_t: float = 0.02,
) -> torch.Tensor:
    """Penalize keeping both feet planted when the clock expects a swing foot."""
    sensor = env.scene.sensors[sensor_cfg.name]
    contact, _ = _foot_contact_and_load(sensor, sensor_cfg.body_ids)
    phases, _, _ = _phase_masks(env, cycle_s, air_ratio, delta_t)
    expected_contact = phases >= air_ratio
    command = env.command_manager.get_command(command_name)
    return gait.unexpected_double_support(
        contact, expected_contact
    ) * gait.command_is_active(command)


def gait_swing_foot_clearance(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.12,
    std: float = 0.04,
    cycle_s: float = 0.85,
    air_ratio: float = 0.38,
    command_name: str = "base_velocity",
    delta_t: float = 0.02,
) -> torch.Tensor:
    """Reward actual swing-foot clearance instead of weight shifting in place."""
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    contact, _ = _foot_contact_and_load(sensor, sensor_cfg.body_ids)
    phases, _, _ = _phase_masks(env, cycle_s, air_ratio, delta_t)
    expected_contact = phases >= air_ratio
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    score = gait.swing_foot_clearance_score(
        foot_height, contact, expected_contact, target_height, std
    )
    command = env.command_manager.get_command(command_name)
    return score * gait.command_is_active(command)


def gait_swing_forward_placement(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    cycle_s: float = 0.85,
    air_ratio: float = 0.38,
    delta_t: float = 0.02,
    min_step_length: float = 0.05,
    max_step_length: float = 0.24,
    std: float = 0.08,
) -> torch.Tensor:
    """Reward command-consistent forward placement late in swing."""
    asset = env.scene[asset_cfg.name]
    foot_delta_w = (
        asset.data.body_pos_w[:, asset_cfg.body_ids] - asset.data.root_pos_w[:, None, :]
    )
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(
        -1, foot_delta_w.shape[1], -1
    )
    foot_pos_b = quat_apply_inverse(
        root_quat_w.reshape(-1, 4), foot_delta_w.reshape(-1, 3)
    ).reshape(foot_delta_w.shape[0], -1, 3)
    phases = gait.foot_phases(env.episode_length_buf, env.step_dt, cycle_s)
    command = env.command_manager.get_command(command_name)
    return gait.swing_command_placement_score(
        foot_pos_b,
        phases,
        command,
        cycle_s=cycle_s,
        air_ratio=air_ratio,
        delta_t=delta_t,
        min_step_length=min_step_length,
        max_step_length=max_step_length,
        std=std,
    )


def gait_stance_velocity(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    cycle_s: float = 0.85,
    air_ratio: float = 0.38,
    delta_t: float = 0.02,
) -> torch.Tensor:
    """Penalize horizontal foot motion while the gait clock expects support."""
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    contact, _ = _foot_contact_and_load(sensor, sensor_cfg.body_ids)
    phases, _, stance_mask = _phase_masks(env, cycle_s, air_ratio, delta_t)
    expected_stance = phases >= air_ratio
    foot_velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(
        torch.linalg.vector_norm(foot_velocity, dim=-1)
        * contact
        * expected_stance
        * stance_mask,
        dim=1,
    )


def feet_too_near(
    env, asset_cfg: SceneEntityCfg, threshold: float = 0.2
) -> torch.Tensor:
    """Penalize feet collapsing into the same lateral/forward point."""
    asset = env.scene[asset_cfg.name]
    positions = asset.data.body_pos_w[:, asset_cfg.body_ids]
    distance = torch.linalg.vector_norm(positions[:, 0] - positions[:, 1], dim=-1)
    return (threshold - distance).clamp_min(0.0)


def feet_y_distance(
    env, asset_cfg: SceneEntityCfg, target: float = 0.30
) -> torch.Tensor:
    """Keep stance width in the body frame and penalize a scissor / cross step.

    World-Y separation is meaningless once the robot yaws. Body-frame |y_L-y_R|
    is the stance width; feet on the same side of the sagittal plane are the
    crossed-leg failure seen in lateral play.
    """
    asset = env.scene[asset_cfg.name]
    pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    rel = pos_w - asset.data.root_pos_w.unsqueeze(1)
    n = rel.shape[0]
    quat = asset.data.root_quat_w.repeat_interleave(2, dim=0)
    pos_b = quat_apply_inverse(quat, rel.reshape(n * 2, 3)).reshape(n, 2, 3)
    left_y = pos_b[:, 0, 1]
    right_y = pos_b[:, 1, 1]
    sep = (left_y - right_y).abs()
    width_err = (sep - target).abs()
    collapse = (0.12 - sep).clamp_min(0.0)
    same_side = ((left_y * right_y) > 0.0).to(sep.dtype)
    return width_err + 3.0 * collapse + 3.5 * same_side


def feet_stumble(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Detect predominantly horizontal impacts; return zero without force data."""
    sensor = env.scene.sensors[sensor_cfg.name]
    force = getattr(sensor.data, "net_forces_w", None)
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    selected = force[:, sensor_cfg.body_ids]
    horizontal = torch.linalg.vector_norm(selected[..., :2], dim=-1)
    vertical = torch.abs(selected[..., 2])
    return (horizontal > 5.0 * vertical.clamp_min(1.0)).any(dim=1).to(force.dtype)


def feet_force(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 500.0,
    max_reward: float = 400.0,
) -> torch.Tensor:
    """Penalize excessive simulated foot load; disable cleanly without force data."""
    load = _averaged_foot_load(env.scene.sensors[sensor_cfg.name], sensor_cfg.body_ids)
    if load is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (load - threshold).clamp(min=0.0, max=max_reward).mean(dim=1)


def ankle_torque_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1
    )


def action_l1(env, action_indices: tuple[int, ...] | list[int]) -> torch.Tensor:
    """Penalize selected policy action magnitudes by stable action-order indices."""
    action = env.action_manager.action[:, list(action_indices)]
    return torch.sum(torch.abs(action), dim=1)
