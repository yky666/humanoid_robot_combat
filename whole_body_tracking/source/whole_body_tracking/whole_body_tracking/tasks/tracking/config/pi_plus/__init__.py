import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments for Pi Plus
##

gym.register(
    id="Tracking-Flat-PiPlus-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PiPlusFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-PiPlus-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PiPlusFlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-PiPlus-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PiPlusFlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatLowFreqPPORunnerCfg",
    },
)