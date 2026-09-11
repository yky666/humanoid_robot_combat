"""Action term that keeps T800 shoulder and elbow targets fixed."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from .guard_action import guard_targets_with_compensation


class FixedGuardJointPositionAction(JointPositionAction):
    cfg: FixedGuardJointPositionActionCfg

    def __init__(self, cfg: FixedGuardJointPositionActionCfg, env):
        super().__init__(cfg, env)
        if len(cfg.guard_joint_names) != len(cfg.guard_joint_position):
            raise ValueError("guard names and positions must have equal length")
        if len(cfg.fixed_joint_names) != len(cfg.fixed_joint_position):
            raise ValueError("fixed names and positions must have equal length")
        if cfg.compensation_limit <= 0.0 or not torch.isfinite(
            torch.as_tensor(cfg.compensation_limit)
        ):
            raise ValueError("compensation_limit must be finite and positive")
        fixed_names = [*cfg.guard_joint_names, *cfg.fixed_joint_names]
        fixed_positions = [*cfg.guard_joint_position, *cfg.fixed_joint_position]
        if len(set(fixed_names)) != len(fixed_names):
            raise ValueError("guard/fixed joint names must be unique")
        action_names = list(self._joint_names)
        self._guard_ids = torch.tensor(
            [action_names.index(name) for name in fixed_names],
            dtype=torch.long,
            device=self.device,
        )
        self._guard_position = torch.tensor(
            fixed_positions, dtype=torch.float32, device=self.device
        )
        if not torch.isfinite(self._guard_position).all():
            raise ValueError("guard position contains non-finite values")
        compensation_names = list(cfg.compensation_joint_names)
        if len(compensation_names) != len(set(compensation_names)):
            raise ValueError("compensation joint names must be unique")
        unknown_compensation = sorted(
            set(compensation_names) - set(cfg.guard_joint_names)
        )
        if unknown_compensation:
            raise ValueError(
                "compensation joints must be guard joints: "
                + ", ".join(unknown_compensation)
            )
        if cfg.stop_command_threshold < 0.0 or not torch.isfinite(
            torch.as_tensor(cfg.stop_command_threshold)
        ):
            raise ValueError("stop_command_threshold must be finite and non-negative")
        self._compensation_action_ids = torch.tensor(
            [action_names.index(name) for name in compensation_names],
            dtype=torch.long,
            device=self.device,
        )
        self._compensation_guard_ids = torch.tensor(
            [list(cfg.guard_joint_names).index(name) for name in compensation_names],
            dtype=torch.long,
            device=self.device,
        )
        stance_names = list(cfg.stance_joint_names)
        stance_position = list(cfg.stance_joint_position)
        if len(stance_names) != len(stance_position):
            raise ValueError("stance names and positions must have equal length")
        if len(set(stance_names)) != len(stance_names):
            raise ValueError("stance joint names must be unique")
        unknown_stance = sorted(set(stance_names) - set(action_names))
        if unknown_stance:
            raise ValueError("stance joints must be action joints: " + ", ".join(unknown_stance))
        self._stance_ids = torch.tensor(
            [action_names.index(name) for name in stance_names],
            dtype=torch.long,
            device=self.device,
        )
        self._stance_position = torch.tensor(
            stance_position, dtype=torch.float32, device=self.device
        )

    def apply_actions(self):
        command = self._env.command_manager.get_command(self.cfg.command_name)
        moving = (
            torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
            > self.cfg.stop_command_threshold
        )
        target = guard_targets_with_compensation(
            self.processed_actions,
            self.raw_actions,
            self._guard_ids,
            self._guard_position,
            self._compensation_action_ids,
            self._compensation_guard_ids,
            self.cfg.compensation_limit,
            moving,
        )
        # Stop / start frames: hold the extracted baoquan crouch on the legs.
        if self._stance_ids.numel():
            stance = self._stance_position.unsqueeze(0).expand(target.shape[0], -1)
            target = target.clone()
            target[:, self._stance_ids] = torch.where(
                moving[:, None], target[:, self._stance_ids], stance
            )
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)


@configclass
class FixedGuardJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = FixedGuardJointPositionAction
    guard_joint_names: list[str] = []  # noqa: RUF012
    guard_joint_position: list[float] = []  # noqa: RUF012
    fixed_joint_names: list[str] = []  # noqa: RUF012
    fixed_joint_position: list[float] = []  # noqa: RUF012
    compensation_joint_names: list[str] = []  # noqa: RUF012
    compensation_limit: float = 0.10
    command_name: str = "base_velocity"
    stop_command_threshold: float = 0.1
    stance_joint_names: list[str] = []  # noqa: RUF012
    stance_joint_position: list[float] = []  # noqa: RUF012
