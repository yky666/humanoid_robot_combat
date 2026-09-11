"""Gym registrations kept separate from the existing tracking registrations."""

import gymnasium as gym

from . import agents, fixed_guard_72_env_cfg, fixed_guard_env_cfg

gym.register(
    id="Tracking-Flat-T800-Fixed-Guard-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fixed_guard_env_cfg.T800FixedGuardVelocityEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FixedGuardVelocityPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Fixed-Guard-72-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fixed_guard_72_env_cfg.T800FixedGuardVelocity72EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FixedGuardVelocity72PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Fixed-Guard-72-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fixed_guard_72_env_cfg.T800FixedGuardVelocity72EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FixedGuardVelocity72PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Fixed-Guard-72-Transition-Safe-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fixed_guard_72_env_cfg.T800FixedGuardVelocity72TransitionSafeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FixedGuardVelocity72PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Fixed-Guard-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fixed_guard_env_cfg.T800FixedGuardVelocityEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FixedGuardVelocityPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Rough-T800-Fixed-Guard-72-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fixed_guard_72_env_cfg.T800FixedGuardVelocity72RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FixedGuardVelocity72PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Rough-T800-Fixed-Guard-72-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fixed_guard_72_env_cfg.T800FixedGuardVelocity72RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FixedGuardVelocity72PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Bump-T800-Fixed-Guard-72-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fixed_guard_72_env_cfg.T800FixedGuardVelocity72BumpEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FixedGuardVelocity72PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Bump-T800-Fixed-Guard-72-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fixed_guard_72_env_cfg.T800FixedGuardVelocity72BumpEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FixedGuardVelocity72PPORunnerCfg",
    },
)
