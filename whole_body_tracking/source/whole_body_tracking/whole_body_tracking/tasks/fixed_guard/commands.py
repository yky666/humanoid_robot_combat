"""Velocity command generation for the fixed-guard gait task."""

from __future__ import annotations

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

from . import curriculum


class FixedGuardVelocityCommand(UniformVelocityCommand):
    """Uniform command sampler with an explicit gap around zero when moving."""

    cfg: FixedGuardVelocityCommandCfg

    def __init__(self, cfg: FixedGuardVelocityCommandCfg, env) -> None:
        super().__init__(cfg, env)
        # The transition wrapper applies a ramped command to ``vel_command_b``
        # while this immutable-within-an-episode buffer retains the sampled
        # target.  This keeps command resampling and the safety handoff
        # independent without adding either value to the actor observation.
        self.requested_vel_command_b = self.vel_command_b.clone()

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids] = curriculum.enforce_minimum_moving_speed(
            self.vel_command_b[env_ids],
            self.is_standing_env[env_ids],
            self.cfg.minimum_moving_speed,
        )
        self.requested_vel_command_b[env_ids] = self.vel_command_b[env_ids]


@configclass
class FixedGuardVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = FixedGuardVelocityCommand
    minimum_moving_speed: float = 0.2
