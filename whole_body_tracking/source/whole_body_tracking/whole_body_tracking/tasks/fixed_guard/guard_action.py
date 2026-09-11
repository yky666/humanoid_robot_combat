"""Framework-independent fixed-guard target construction helpers."""

from __future__ import annotations

import torch


def guard_targets_with_compensation(
    processed_actions: torch.Tensor,
    raw_actions: torch.Tensor,
    fixed_ids: torch.Tensor,
    fixed_position: torch.Tensor,
    compensation_action_ids: torch.Tensor,
    compensation_guard_ids: torch.Tensor,
    compensation_limit: float,
    moving: torch.Tensor,
) -> torch.Tensor:
    """Build fixed-guard targets with a bounded locomotion-only arm residual."""
    if processed_actions.ndim != 2 or raw_actions.shape != processed_actions.shape:
        raise ValueError(
            "processed_actions and raw_actions must have matching [env, action] shape"
        )
    if (
        fixed_ids.ndim != 1
        or fixed_position.ndim != 1
        or fixed_ids.numel() != fixed_position.numel()
    ):
        raise ValueError(
            "fixed guard ids and positions must have matching one-dimensional shapes"
        )
    if compensation_action_ids.numel() != compensation_guard_ids.numel():
        raise ValueError("compensation action and guard ids must have equal length")
    if compensation_limit <= 0.0 or not torch.isfinite(
        torch.as_tensor(compensation_limit)
    ):
        raise ValueError("compensation_limit must be finite and positive")
    if moving.ndim != 1 or moving.shape[0] != processed_actions.shape[0]:
        raise ValueError("moving must have one boolean value per environment")

    target = processed_actions.clone()
    target[:, fixed_ids] = fixed_position
    if compensation_action_ids.numel():
        residual = raw_actions[:, compensation_action_ids].clamp(
            min=-compensation_limit, max=compensation_limit
        )
        guard_target = fixed_position[compensation_guard_ids]
        target[:, compensation_action_ids] = guard_target + residual * moving[:, None]
    return target
