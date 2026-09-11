from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from whole_body_tracking.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg import (
    T800FlatPPORunnerCfg,
)


@configclass
class T800FixedGuardVelocityPPORunnerCfg(T800FlatPPORunnerCfg):
    """PPO defaults for the online velocity task (observation size is inferred)."""

    experiment_name = "t800_fixed_guard_velocity"
    max_iterations = 30000
    # This task intentionally exposes one policy observation group only.
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}  # noqa: RUF012
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class T800FixedGuardVelocity72PPORunnerCfg(T800FixedGuardVelocityPPORunnerCfg):
    """Separate log namespace for the SDK-compatible 72-D/22-D retrain."""

    experiment_name = "t800_fixed_guard_velocity_72d"
