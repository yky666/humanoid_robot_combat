from __future__ import annotations

import copy
from dataclasses import MISSING
from typing import Any


def _is_missing(value: Any) -> bool:
    return value is MISSING or value.__class__.__name__ == "_MISSING_TYPE"


def _value(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    value = mapping.get(key, default)
    return default if _is_missing(value) else value


def adapt_legacy_ppo_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Adapt IsaacLab actor-critic PPO config to rsl_rl>=3 actor/critic config."""
    cfg = copy.deepcopy(train_cfg)
    if "actor" in cfg and "critic" in cfg:
        return cfg

    policy = cfg.pop("policy", None)
    if policy is None:
        return cfg

    empirical_norm = bool(_value(cfg, "empirical_normalization", False))
    model_class = "RNNModel" if "Recurrent" in str(_value(policy, "class_name", "")) else "MLPModel"
    distribution_class = (
        "HeteroscedasticGaussianDistribution" if bool(_value(policy, "state_dependent_std", False)) else "GaussianDistribution"
    )

    actor = {
        "class_name": model_class,
        "hidden_dims": _value(policy, "actor_hidden_dims", [256, 256, 256]),
        "activation": _value(policy, "activation", "elu"),
        "obs_normalization": bool(_value(policy, "actor_obs_normalization", empirical_norm)),
        "distribution_cfg": {
            "class_name": distribution_class,
            "init_std": float(_value(policy, "init_noise_std", 1.0)),
            "std_type": _value(policy, "noise_std_type", "scalar"),
        },
    }
    critic = {
        "class_name": model_class,
        "hidden_dims": _value(policy, "critic_hidden_dims", [256, 256, 256]),
        "activation": _value(policy, "activation", "elu"),
        "obs_normalization": bool(_value(policy, "critic_obs_normalization", empirical_norm)),
    }

    if model_class == "RNNModel":
        for target in (actor, critic):
            target["rnn_type"] = _value(policy, "rnn_type", "lstm")
            target["rnn_hidden_dim"] = _value(policy, "rnn_hidden_dim", 256)
            target["rnn_num_layers"] = _value(policy, "rnn_num_layers", 1)

    algorithm = cfg.setdefault("algorithm", {})
    algorithm["class_name"] = _value(algorithm, "class_name", "PPO")
    algorithm["rnd_cfg"] = None if _is_missing(algorithm.get("rnd_cfg")) else algorithm.get("rnd_cfg")
    algorithm["symmetry_cfg"] = None if _is_missing(algorithm.get("symmetry_cfg")) else algorithm.get("symmetry_cfg")

    obs_groups = cfg.get("obs_groups")
    if _is_missing(obs_groups) or not obs_groups:
        cfg["obs_groups"] = {"actor": ["policy"], "critic": ["critic"]}

    if _is_missing(cfg.get("clip_actions")):
        cfg["clip_actions"] = None

    cfg["actor"] = actor
    cfg["critic"] = critic
    return cfg


def get_policy_module(algorithm: Any) -> Any:
    if hasattr(algorithm, "get_policy"):
        return algorithm.get_policy()
    if hasattr(algorithm, "policy"):
        return algorithm.policy
    if hasattr(algorithm, "actor"):
        return algorithm.actor
    raise AttributeError("Could not locate a policy module on the rsl_rl algorithm.")
