"""Pure, deployment-aligned helpers for a safe stand-to-walk handoff.

These helpers deliberately do not change the actor observation or action
contract.  They are shared by the 72-D training configuration and its tests so
that the deployment profile's alpha, command ramp, action-history seed, and
per-step target limit have a small, auditable reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TransitionRandomization:
    """Auditable per-episode randomization sampled by the training wrapper."""

    seed: int
    mass_scale: float
    com_offset: tuple[float, float, float]
    initial_linear_velocity: tuple[float, float, float]
    initial_angular_velocity: tuple[float, float, float]
    action_delay_steps: int
    activation_time_s: float
    push_direction: tuple[float, float]
    push_magnitude: float
    push_phase_s: float


def sample_randomization(
    seed: int,
    *,
    action_delay_steps: tuple[int, int] = (0, 2),
    activation_time_s: tuple[float, float] = (0.0, 1.0),
) -> TransitionRandomization:
    """Sample reproducible, bounded transition disturbances for episode logs."""
    if action_delay_steps[0] < 0 or action_delay_steps[1] < action_delay_steps[0]:
        raise ValueError("invalid action delay range")
    if activation_time_s[0] < 0.0 or activation_time_s[1] < activation_time_s[0]:
        raise ValueError("invalid activation time range")
    generator = torch.Generator().manual_seed(seed)

    def uniform(low: float, high: float) -> float:
        return float(torch.empty((), dtype=torch.float32).uniform_(low, high, generator=generator))

    return TransitionRandomization(
        seed=seed,
        mass_scale=uniform(0.95, 1.05),
        com_offset=tuple(uniform(-0.015, 0.015) for _ in range(3)),
        initial_linear_velocity=tuple(uniform(-0.05, 0.05) for _ in range(3)),
        initial_angular_velocity=tuple(uniform(-0.10, 0.10) for _ in range(3)),
        action_delay_steps=int(torch.randint(action_delay_steps[0], action_delay_steps[1] + 1, (), generator=generator)),
        activation_time_s=uniform(*activation_time_s),
        push_direction=(uniform(-1.0, 1.0), uniform(-1.0, 1.0)),
        push_magnitude=uniform(0.05, 0.25),
        push_phase_s=uniform(0.0, 15.0),
    )


def transition_alpha(elapsed_s: torch.Tensor | float, duration_s: float) -> torch.Tensor:
    """Return finite, bounded, monotonic handoff alpha values in ``[0, 1]``."""
    if duration_s <= 0.0 or not torch.isfinite(torch.tensor(duration_s)):
        raise ValueError("duration_s must be finite and positive")
    elapsed_s = torch.as_tensor(elapsed_s, dtype=torch.float32)
    if not torch.isfinite(elapsed_s).all():
        raise ValueError("elapsed_s contains non-finite values")
    return (elapsed_s / duration_s).clamp_(0.0, 1.0)


def ramp_command(
    current: torch.Tensor, requested: torch.Tensor, max_delta: torch.Tensor | float
) -> torch.Tensor:
    """Move a command toward its request without an acceleration/stop jump."""
    if current.shape != requested.shape:
        raise ValueError("current and requested commands must have the same shape")
    delta = torch.as_tensor(max_delta, dtype=current.dtype, device=current.device)
    if not torch.isfinite(current).all() or not torch.isfinite(requested).all():
        raise ValueError("commands must be finite")
    if not torch.isfinite(delta).all() or torch.any(delta <= 0.0):
        raise ValueError("max_delta must be finite and positive")
    return current + (requested - current).clamp(min=-delta, max=delta)


def limit_target_delta(
    previous: torch.Tensor, requested: torch.Tensor, max_delta: torch.Tensor | float
) -> torch.Tensor:
    """Bound every joint target change while preserving tensor shape/order."""
    return ramp_command(previous, requested, max_delta)


def blend_targets(
    hold_target: torch.Tensor, policy_target: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Blend a held stand target into a policy target without new actor fields."""
    if hold_target.shape != policy_target.shape:
        raise ValueError("hold_target and policy_target must have the same shape")
    if not torch.isfinite(hold_target).all() or not torch.isfinite(policy_target).all():
        raise ValueError("targets must be finite")
    alpha = torch.as_tensor(alpha, dtype=hold_target.dtype, device=hold_target.device)
    if not torch.isfinite(alpha).all() or torch.any((alpha < 0.0) | (alpha > 1.0)):
        raise ValueError("alpha must be finite and in [0, 1]")
    while alpha.ndim < hold_target.ndim:
        alpha = alpha.unsqueeze(-1)
    return torch.lerp(hold_target, policy_target, alpha)


def seed_history(observation: torch.Tensor, history_length: int) -> torch.Tensor:
    """Repeat the current 72-D frame across the SDK-compatible 15-frame seed."""
    if observation.ndim < 1 or history_length <= 0:
        raise ValueError("observation must be non-empty and history_length positive")
    if not torch.isfinite(observation).all():
        raise ValueError("observation contains non-finite values")
    return observation.unsqueeze(-2).repeat_interleave(history_length, dim=-2)


def seed_action_history(action_dim: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return the explicit zero action seed used at a transition boundary."""
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    return torch.zeros(action_dim, dtype=torch.float32, device=device)


class TransitionController:
    """Vectorized stand/command handoff used at the training action boundary."""

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        *,
        device: torch.device | str = "cpu",
        duration_s: float = 0.5,
        command_ramp_rate: tuple[float, float, float] = (1.0, 0.6, 1.5),
        max_target_delta: float = 0.03,
        max_delay_steps: int = 2,
        activation_time_s: tuple[float, float] = (0.0, 1.0),
        seed: int = 42,
    ) -> None:
        if num_envs <= 0 or action_dim <= 0:
            raise ValueError("num_envs and action_dim must be positive")
        if duration_s <= 0.0 or max_target_delta <= 0.0 or max_delay_steps < 0:
            raise ValueError(
                "duration_s and max_target_delta must be positive; max_delay_steps non-negative"
            )
        if activation_time_s[0] < 0.0 or activation_time_s[1] < activation_time_s[0]:
            raise ValueError("activation_time_s must be a non-negative ordered range")
        rates = torch.as_tensor(command_ramp_rate, dtype=torch.float32, device=device)
        if rates.shape != (3,) or not torch.isfinite(rates).all() or torch.any(rates <= 0):
            raise ValueError("command_ramp_rate must contain three positive values")
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.duration_s = float(duration_s)
        self.command_ramp_rate = rates
        self.max_target_delta = float(max_target_delta)
        self.max_delay_steps = int(max_delay_steps)
        self.activation_time_s = activation_time_s
        self._generator = torch.Generator(device=self.device).manual_seed(seed)
        self.elapsed_s = torch.zeros(num_envs, device=self.device)
        self.hold_action = torch.zeros((num_envs, action_dim), device=self.device)
        self.previous_action = self.hold_action.clone()
        self.command = torch.zeros((num_envs, 3), device=self.device)
        self.delay_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.activation_s = torch.zeros(num_envs, device=self.device)
        self._delay_buffer = torch.zeros(
            (max_delay_steps + 1, num_envs, action_dim), device=self.device
        )
        self._sample_episode_parameters(torch.arange(num_envs, device=self.device))

    def _sample_episode_parameters(self, ids: torch.Tensor) -> None:
        self.delay_steps[ids] = torch.randint(
            0,
            self.max_delay_steps + 1,
            (len(ids),),
            generator=self._generator,
            device=self.device,
        )
        self.activation_s[ids] = torch.empty(
            len(ids), device=self.device
        ).uniform_(
            self.activation_time_s[0],
            self.activation_time_s[1],
            generator=self._generator,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        )
        self.elapsed_s[ids] = 0.0
        self.hold_action[ids] = 0.0
        self.previous_action[ids] = 0.0
        self.command[ids] = 0.0
        self._delay_buffer[:, ids] = 0.0
        self._sample_episode_parameters(ids)

    def step(
        self,
        policy_action: torch.Tensor,
        requested_command: torch.Tensor,
        dt: float,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if policy_action.shape != self.previous_action.shape:
            raise ValueError("policy_action shape does not match action_dim")
        if requested_command.shape != self.command.shape:
            raise ValueError("requested_command must have shape [num_envs, 3]")
        if reset_mask is not None and torch.any(reset_mask):
            self.reset(torch.nonzero(reset_mask, as_tuple=False).squeeze(-1))
        active_elapsed = (self.elapsed_s - self.activation_s).clamp_min(0.0)
        alpha = transition_alpha(active_elapsed, self.duration_s).to(policy_action.device)
        bounded = blend_targets(self.hold_action, policy_action, alpha)
        bounded = limit_target_delta(self.previous_action, bounded, self.max_target_delta)
        self._delay_buffer = torch.cat((bounded.unsqueeze(0), self._delay_buffer[:-1]), dim=0)
        delayed = self._delay_buffer[self.delay_steps, torch.arange(self.num_envs, device=self.device)]
        # ``elapsed_s`` advances in float32 policy steps.  Permit one machine
        # epsilon at the activation boundary so 0.02 s steps reach 0.10 s
        # deterministically instead of waiting an accidental extra frame.
        active = (
            self.elapsed_s + torch.finfo(self.elapsed_s.dtype).eps >= self.activation_s
        ).unsqueeze(-1)
        active_request = torch.where(active, requested_command, torch.zeros_like(requested_command))
        command_delta = (active_request - self.command).clamp(
            min=-(self.command_ramp_rate * dt), max=self.command_ramp_rate * dt
        )
        self.command = self.command + command_delta
        self.previous_action = bounded.detach()
        self.elapsed_s += float(dt)
        return delayed, self.command.clone(), alpha
