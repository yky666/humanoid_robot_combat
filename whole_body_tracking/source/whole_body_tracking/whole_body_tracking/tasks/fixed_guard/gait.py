"""Pure gait-shaping helpers for the fixed-guard locomotion task."""

from __future__ import annotations

import math

import torch


def foot_phases(
    episode_length: torch.Tensor,
    step_dt: float,
    cycle_s: float = 0.85,
    phase_offsets: tuple[float, float] = (0.38, 0.88),
) -> torch.Tensor:
    """Return TienKung-compatible left/right phases in ``[0, 1)``."""
    if cycle_s <= 0.0:
        raise ValueError("cycle_s must be positive")
    if len(phase_offsets) != 2:
        raise ValueError("phase_offsets must contain left and right values")
    step_time = episode_length.to(dtype=torch.float32) * step_dt
    offsets = torch.as_tensor(
        phase_offsets, dtype=step_time.dtype, device=episode_length.device
    )
    return torch.remainder(step_time[:, None] / cycle_s + offsets, 1.0)


def phase_features(
    episode_length: torch.Tensor,
    step_dt: float,
    cycle_s: float = 0.85,
) -> torch.Tensor:
    """Encode both foot clocks so the policy can observe the reward phase."""
    angles = foot_phases(episode_length, step_dt, cycle_s) * (2.0 * math.pi)
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)


def gait_clock(
    phase: torch.Tensor,
    air_ratio: float = 0.38,
    delta_t: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return smooth swing/support masks matching TienKung's gait clock."""
    if not 0.0 < air_ratio < 1.0:
        raise ValueError("air_ratio must be in (0, 1)")
    if delta_t <= 0.0 or delta_t >= air_ratio / 2.0:
        raise ValueError("delta_t must be positive and smaller than half air_ratio")
    swing = (phase >= delta_t) & (phase <= air_ratio - delta_t)
    transition_start = phase < delta_t
    transition_air = (phase > air_ratio - delta_t) & (phase < air_ratio + delta_t)
    transition_end = phase > 1.0 - delta_t
    swing_mask = (
        swing.to(dtype=phase.dtype)
        + (0.5 + phase / (2.0 * delta_t)) * transition_start
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * transition_air
        + (phase - 1.0 + delta_t) / (2.0 * delta_t) * transition_end
    ).clamp(0.0, 1.0)
    return swing_mask, 1.0 - swing_mask


def expected_foot_contacts(
    episode_length: torch.Tensor,
    step_dt: float,
    cycle_s: float = 0.85,
    air_ratio: float = 0.38,
) -> torch.Tensor:
    """Return the desired left/right contact pattern for each environment."""
    return foot_phases(episode_length, step_dt, cycle_s) >= air_ratio


def command_is_active(command: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Select commands that require locomotion rather than standing."""
    return (
        torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        > threshold
    )


def exact_contact_pattern_score(
    contact: torch.Tensor, expected_contact: torch.Tensor
) -> torch.Tensor:
    """Reward only a complete match so double support cannot earn half credit."""
    return torch.all(contact == expected_contact, dim=1).to(dtype=torch.float32)


def unexpected_double_support(
    contact: torch.Tensor, expected_contact: torch.Tensor
) -> torch.Tensor:
    """Flag double support only while the gait clock expects a swing foot."""
    return (torch.all(contact, dim=1) & ~torch.all(expected_contact, dim=1)).to(
        dtype=torch.float32
    )


def swing_foot_clearance_score(
    foot_height: torch.Tensor,
    contact: torch.Tensor,
    expected_contact: torch.Tensor,
    target_height: float = 0.12,
    std: float = 0.04,
) -> torch.Tensor:
    """Reward a non-contact swing foot near the requested ankle-link height."""
    expected_swing = ~expected_contact
    actual_swing = expected_swing & ~contact
    height_score = torch.exp(-torch.square((foot_height - target_height) / std))
    swing_count = expected_swing.sum(dim=1).clamp_min(1)
    return (height_score * actual_swing).sum(dim=1) / swing_count


def swing_forward_placement_score(
    foot_pos_b: torch.Tensor,
    phase: torch.Tensor,
    command: torch.Tensor,
    cycle_s: float = 0.85,
    air_ratio: float = 0.38,
    delta_t: float = 0.02,
    min_step_length: float = 0.05,
    max_step_length: float = 0.24,
    std: float = 0.08,
) -> torch.Tensor:
    """Reward late-swing placement along the commanded translational direction.

    Despite the historical name, this handles forward, backward, and lateral
    commands.  The swing foot is compared with the opposite stance foot and
    projected onto the normalized ``(vx, vy)`` direction.  Pure-yaw commands
    bypass the term because they do not define a valid linear step target.
    """
    if foot_pos_b.ndim != 3 or foot_pos_b.shape[1:] != (2, 3):
        raise ValueError("foot_pos_b must have shape [num_envs, 2, 3]")
    if phase.shape != foot_pos_b.shape[:2]:
        raise ValueError("phase must have shape [num_envs, 2]")
    if (
        command.ndim != 2
        or command.shape[0] != foot_pos_b.shape[0]
        or command.shape[1] < 3
    ):
        raise ValueError("command must have shape [num_envs, 3]")
    if cycle_s <= 0.0 or not 0.0 < air_ratio < 1.0:
        raise ValueError("cycle_s must be positive and air_ratio must be in (0, 1)")
    if delta_t <= 0.0 or delta_t >= air_ratio / 2.0:
        raise ValueError("delta_t must be positive and smaller than half air_ratio")
    if min_step_length < 0.0 or max_step_length < min_step_length or std <= 0.0:
        raise ValueError("invalid step-length or std parameters")

    swing_mask, _ = gait_clock(phase, air_ratio, delta_t)
    swing_progress = ((phase - delta_t) / (air_ratio - 2.0 * delta_t)).clamp(0.0, 1.0)
    late = ((swing_progress - 0.55) / 0.30).clamp(0.0, 1.0)
    late = late * late * (3.0 - 2.0 * late)
    weights = swing_mask * late

    command_xy = command[:, :2]
    translation_speed = torch.linalg.vector_norm(command_xy, dim=1)
    translation_dominant = command_is_active(command) & (
        translation_speed >= 0.25 * torch.abs(command[:, 2])
    )
    direction = command_xy / translation_speed.clamp_min(1.0e-6).unsqueeze(1)
    target = (translation_speed * cycle_s * 0.5).clamp(
        min=min_step_length, max=max_step_length
    )
    opposite_xy = foot_pos_b[:, [1, 0], :2]
    relative_xy = foot_pos_b[:, :, :2] - opposite_xy
    relative_step = torch.sum(relative_xy * direction.unsqueeze(1), dim=-1)
    score = torch.exp(-torch.square((relative_step - target.unsqueeze(1)) / std))
    score = (score * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0e-6)
    return score * translation_dominant.to(dtype=score.dtype)


def swing_command_placement_score(*args, **kwargs) -> torch.Tensor:
    """Descriptive alias for the command-aligned placement score."""
    return swing_forward_placement_score(*args, **kwargs)


def normalized_foot_force(
    force: torch.Tensor,
    force_scale: float,
    max_normalized_force: float = 2.0,
) -> torch.Tensor:
    """Convert averaged foot-force magnitudes to a bounded body-weight scale."""
    if force_scale <= 0.0 or max_normalized_force <= 0.0:
        raise ValueError("force scales must be positive")
    return (force / force_scale).clamp(min=0.0, max=max_normalized_force)


def swing_force_score(
    normalized_force: torch.Tensor,
    expected_contact: torch.Tensor,
    force_std: float = 0.08,
) -> torch.Tensor:
    """Reward low load on the foot selected by the gait clock to swing."""
    if force_std <= 0.0:
        raise ValueError("force_std must be positive")
    expected_swing = ~expected_contact
    score = torch.exp(-torch.square(normalized_force / force_std))
    return (score * expected_swing).sum(dim=1) / expected_swing.sum(dim=1).clamp_min(1)


def smooth_swing_force_score(
    normalized_force: torch.Tensor,
    swing_mask: torch.Tensor,
    force_std: float = 0.08,
) -> torch.Tensor:
    """Reward low load under a continuous swing-phase mask."""
    if force_std <= 0.0:
        raise ValueError("force_std must be positive")
    score = torch.exp(-torch.square(normalized_force / force_std))
    return (score * swing_mask).sum(dim=1) / swing_mask.sum(dim=1).clamp_min(1.0)


def stance_velocity_score(
    foot_speed: torch.Tensor,
    expected_contact: torch.Tensor,
    speed_std: float = 0.15,
) -> torch.Tensor:
    """Reward low horizontal velocity for the expected stance foot."""
    if speed_std <= 0.0:
        raise ValueError("speed_std must be positive")
    score = torch.exp(-torch.square(foot_speed / speed_std))
    return (score * expected_contact).sum(dim=1) / expected_contact.sum(
        dim=1
    ).clamp_min(1)


def smooth_stance_velocity_score(
    foot_speed: torch.Tensor,
    stance_mask: torch.Tensor,
    speed_std: float = 0.15,
) -> torch.Tensor:
    """Reward low stance-foot speed under a continuous support mask."""
    if speed_std <= 0.0:
        raise ValueError("speed_std must be positive")
    score = torch.exp(-torch.square(foot_speed / speed_std))
    return (score * stance_mask).sum(dim=1) / stance_mask.sum(dim=1).clamp_min(1.0)


def stance_support_force_score(
    normalized_force: torch.Tensor,
    expected_contact: torch.Tensor,
    force_std: float = 0.15,
) -> torch.Tensor:
    """Reward nonzero support load on the expected stance foot."""
    if force_std <= 0.0:
        raise ValueError("force_std must be positive")
    score = 1.0 - torch.exp(-torch.square(normalized_force / force_std))
    return (score * expected_contact).sum(dim=1) / expected_contact.sum(
        dim=1
    ).clamp_min(1)


def smooth_stance_support_force_score(
    normalized_force: torch.Tensor,
    stance_mask: torch.Tensor,
    force_std: float = 0.15,
) -> torch.Tensor:
    """Reward nonzero stance load under a continuous support mask."""
    if force_std <= 0.0:
        raise ValueError("force_std must be positive")
    score = 1.0 - torch.exp(-torch.square(normalized_force / force_std))
    return (score * stance_mask).sum(dim=1) / stance_mask.sum(dim=1).clamp_min(1.0)


def smooth_binary_swing_score(
    contact: torch.Tensor, swing_mask: torch.Tensor
) -> torch.Tensor:
    """Fallback swing score when no force magnitude is available."""
    return ((~contact).to(dtype=swing_mask.dtype) * swing_mask).sum(
        dim=1
    ) / swing_mask.sum(dim=1).clamp_min(1.0)


def smooth_binary_stance_score(
    contact: torch.Tensor, stance_mask: torch.Tensor
) -> torch.Tensor:
    """Fallback support score when no force magnitude is available."""
    return (contact.to(dtype=stance_mask.dtype) * stance_mask).sum(
        dim=1
    ) / stance_mask.sum(dim=1).clamp_min(1.0)


def gate_tracking_reward(
    tracking_reward: torch.Tensor,
    gait_compliance: torch.Tensor,
    command_active: torch.Tensor,
    gate_floor: float = 0.2,
) -> torch.Tensor:
    """Limit moving-command tracking credit until the contact cycle is real."""
    if not 0.0 <= gate_floor <= 1.0:
        raise ValueError("gate_floor must be in [0, 1]")
    gate = gate_floor + (1.0 - gate_floor) * gait_compliance.clamp(0.0, 1.0)
    return torch.where(command_active, tracking_reward * gate, tracking_reward)
