import os

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.utils.rsl_rl_compat import adapt_legacy_ppo_cfg, get_policy_module


def _logger_type(runner: OnPolicyRunner) -> str:
    logger_type = getattr(runner, "logger_type", None) or getattr(getattr(runner, "logger", None), "logger_type", None)
    return (logger_type or runner.cfg.get("logger", "tensorboard")).lower()


def _adapt_distribution_state_dict(loaded_dict: dict, runner: OnPolicyRunner) -> tuple[dict | None, bool]:
    actor_state = loaded_dict.get("actor_state_dict")
    if actor_state is None:
        return None, False

    target_state = runner.alg.actor.state_dict()
    converted = False
    if "distribution.log_std_param" in target_state and "distribution.std_param" in actor_state:
        std = actor_state.pop("distribution.std_param").detach().clone().clamp_min(1.0e-6)
        actor_state["distribution.log_std_param"] = torch.log(std)
        converted = True
    elif "distribution.std_param" in target_state and "distribution.log_std_param" in actor_state:
        log_std = actor_state.pop("distribution.log_std_param").detach().clone()
        actor_state["distribution.std_param"] = torch.exp(log_std).clamp_min(1.0e-6)
        converted = True

    load_cfg = None
    if converted:
        load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": True}
        print("[INFO] Converted Gaussian std checkpoint parameterization; skipping optimizer state reload.")
    return load_cfg, converted


class MyOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, adapt_legacy_ppo_cfg(train_cfg), log_dir, device)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        converted_load_cfg, converted = _adapt_distribution_state_dict(loaded_dict, self)
        if converted and load_cfg is None:
            load_cfg = converted_load_cfg
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if _logger_type(self) in ["wandb"]:
            import wandb

            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            self.export_policy_to_onnx(policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, adapt_legacy_ppo_cfg(train_cfg), log_dir, device)
        self.registry_name = registry_name

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        converted_load_cfg, converted = _adapt_distribution_state_dict(loaded_dict, self)
        if converted and load_cfg is None:
            load_cfg = converted_load_cfg
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if _logger_type(self) in ["wandb"]:
            import wandb

            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                get_policy_module(self.alg),
                normalizer=getattr(self, "obs_normalizer", None),
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                if ":" in self.registry_name:
                    wandb.run.use_artifact(self.registry_name)
                else:
                    print(
                        f"[WARN] Skipping wandb.use_artifact for registry_name='{self.registry_name}'. "
                        "Expected format 'collection:alias'."
                    )
                self.registry_name = None
