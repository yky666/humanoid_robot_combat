from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
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
