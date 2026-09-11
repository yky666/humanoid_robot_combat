"""Observation adapters shared by SDK-compatible training and playback."""

from __future__ import annotations

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from tensordict import TensorDict

from whole_body_tracking.tasks.fixed_guard.transition import (
    TransitionController,
    sample_randomization,
)


class Sdk72CommandTailWrapper(gym.Wrapper):
    """Append the velocity command after a 15-frame 72-D observation history."""

    _COMMAND_SCALE = (2.0, 2.0, 1.0)

    def _augment(self, obs):
        if not isinstance(obs, dict) or "policy" not in obs:
            raise TypeError("SDK 72-D task requires a dict observation with a policy group")
        command = self.env.unwrapped.command_manager.get_command("base_velocity")[:, :3]
        command = command.to(device=obs["policy"].device, dtype=obs["policy"].dtype)
        command = command * torch.as_tensor(self._COMMAND_SCALE, device=command.device, dtype=command.dtype)
        updated = dict(obs)
        updated["policy"] = torch.cat((obs["policy"], command), dim=-1)
        return updated

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    def get_observations(self):
        """Return the command-tail augmented observation for RSL-RL startup."""
        base = self.env.unwrapped
        if hasattr(base, "observation_manager"):
            obs = base.observation_manager.compute()
        elif hasattr(self.env, "get_observations"):
            obs = self.env.get_observations()
        else:
            obs = base._get_observations()
        return self._augment(obs)


class TransitionSafeWrapper(gym.Wrapper):
    """Apply the training-side stand/command handoff without changing actor dims."""

    def __init__(
        self,
        env,
        *,
        duration_s=0.5,
        max_target_delta=0.03,
        command_ramp_rate=(1.0, 0.6, 1.5),
        max_delay_steps=2,
        activation_time_s=(0.0, 1.0),
        seed=42,
    ):
        super().__init__(env)
        base = env.unwrapped
        self._controller = TransitionController(
            int(base.num_envs),
            int(env.action_space.shape[-1]),
            device=getattr(base, "device", "cpu"),
            duration_s=duration_s,
            command_ramp_rate=command_ramp_rate,
            max_target_delta=max_target_delta,
            max_delay_steps=max_delay_steps,
            activation_time_s=activation_time_s,
            seed=seed,
        )
        self.transition_diagnostics = {}
        self._seed = int(seed)
        self._episode_index = 0
        self.transition_randomization = sample_randomization(self._seed)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._controller.reset()
        self._set_active_command(self._controller.command)
        self.transition_randomization = sample_randomization(self._seed + self._episode_index)
        self._episode_index += 1
        return self._with_amp_features(obs, self._controller.command), info

    def step(self, action):
        base = self.env.unwrapped
        command_term = base.command_manager.get_term("base_velocity")
        requested = getattr(command_term, "requested_vel_command_b", command_term.vel_command_b)[:, :3]
        dt = float(getattr(base, "step_dt", 0.02))
        reset_mask = getattr(base, "reset_buf", None)
        previous_action = self._controller.previous_action.clone()
        bounded_action, command, alpha = self._controller.step(action, requested, dt, reset_mask)
        # Keep command observations consistent with the command actually used
        # by the policy. IsaacLab's command term exposes vel_command_b; the
        # fallback is intentionally a no-op for light-weight test doubles.
        self._set_active_command(command)
        obs, reward, terminated, truncated, info = self.env.step(bounded_action)
        dones = terminated | truncated
        if torch.any(dones):
            self._controller.reset(torch.nonzero(dones, as_tuple=False).squeeze(-1))
            command = self._controller.command.clone()
        # CommandManager may have resampled during ``env.step``.  Restore the
        # effective ramped value for rewards, the next policy observation and
        # action compensation; the sampled target remains in its side buffer.
        self._set_active_command(command)
        self.transition_diagnostics = {
            "transition_alpha": alpha.detach(),
            "transition_command": command.detach(),
            "transition_action_delta": (bounded_action - previous_action).detach(),
            "transition_seed": self.transition_randomization.seed,
            "transition_action_delay": self._controller.delay_steps.detach(),
            "transition_activation_time": self._controller.activation_s.detach(),
        }
        if hasattr(base, "extras"):
            base.extras.update(
                {
                    "transition_alpha_mean": alpha.mean().detach(),
                    "transition_command_speed": command[:, :2].norm(dim=-1).mean().detach(),
                    "transition_action_delay_mean": self._controller.delay_steps.float().mean().detach(),
                    "transition_activation_time_mean": self._controller.activation_s.mean().detach(),
                    "transition_random_seed": float(self.transition_randomization.seed),
                }
            )
        return self._with_amp_features(obs, command), reward, terminated, truncated, info

    def _set_active_command(self, command):
        """Write the controller's effective command without losing its target."""
        command_term = self.env.unwrapped.command_manager.get_term("base_velocity")
        if hasattr(command_term, "vel_command_b"):
            command_term.vel_command_b[:, :3] = command

    def get_observations(self):
        """Return AMP side channels when RSL-RL asks before its first step."""
        base = self.env.unwrapped
        if hasattr(base, "observation_manager"):
            obs = base.observation_manager.compute()
        else:
            obs = base._get_observations()
        return self._with_amp_features(obs, self._controller.command)

    def _with_amp_features(self, obs, command):
        """Expose simulator-only lower-body features to AMP, never to the actor."""
        import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
        from isaaclab.managers import SceneEntityCfg

        from whole_body_tracking.tasks.fixed_guard import rewards

        base = self.env.unwrapped
        if not hasattr(self, "_amp_joint_cfg"):
            self._amp_joint_cfg = SceneEntityCfg(
                "robot",
                joint_names=[
                    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
                    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
                    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
                    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
                ],
                preserve_order=True,
            )
            self._amp_foot_cfg = SceneEntityCfg(
                "robot",
                body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"],
                preserve_order=True,
            )
            # Observation manager normally resolves selectors during setup.
            # This AMP-only path sits outside that manager, so do it once.
            self._amp_joint_cfg.resolve(base.scene)
            self._amp_foot_cfg.resolve(base.scene)
        amp = torch.cat(
            (
                velocity_mdp.joint_pos_rel(base, self._amp_joint_cfg),
                velocity_mdp.joint_vel_rel(base, self._amp_joint_cfg),
                rewards.feet_positions_b(base, self._amp_foot_cfg),
            ),
            dim=-1,
        )
        updated = dict(obs)
        updated["amp"] = amp
        updated["amp_command"] = command.to(device=amp.device, dtype=amp.dtype)
        return updated


class SdkRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL adapter that preserves training-side Gym observation adapters.

    IsaacLab's stock wrapper recomputes observations directly from the base
    environment at runner initialization.  This adapter intentionally calls
    the outer Gym wrapper, so the 72-D command tail and optional AMP-only
    side channels are present both before and after the first environment step.
    """

    def get_observations(self) -> TensorDict:
        if not hasattr(self.env, "get_observations"):
            return super().get_observations()
        obs = self.env.get_observations()
        return TensorDict(obs, batch_size=[self.num_envs])
