import gymnasium as gym

from . import agents, flat_env_cfg

gym.register(
    id="Tracking-Flat-T800-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Recovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800RecoveryEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Getup-Direct-T800-Prone-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800DirectGetupProneEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Getup-Direct-T800-Supine-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800DirectGetupSupineEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Getup-Direct-T800-Mixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800DirectGetupEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Getup-Direct-T800-Prone-Shaped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800DirectGetupProneShapedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Getup-Direct-T800-Supine-Shaped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800DirectGetupSupineShapedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Getup-Direct-T800-Mixed-Shaped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800DirectGetupShapedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Getup-Direct-T800-Prone-Staged-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800DirectGetupProneStagedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Getup-Direct-T800-Supine-Staged-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800DirectGetupSupineStagedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Getup-Direct-T800-Mixed-Staged-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800DirectGetupStagedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Getup-Direct-T800-Prone-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800DirectGetupProneCurriculumEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Getup-Direct-T800-Supine-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800DirectGetupSupineCurriculumEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Getup-Direct-T800-Mixed-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800DirectGetupCurriculumEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T800FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatLowFreqPPORunnerCfg",
    },
)
