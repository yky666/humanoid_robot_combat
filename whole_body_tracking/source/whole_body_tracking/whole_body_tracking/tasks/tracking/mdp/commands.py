from __future__ import annotations

import json
import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int],
        joint_names: Sequence[str],
        legacy_joint_names: Sequence[str],
        device: str = "cpu",
    ):
        self.motion_files = self._resolve_motion_files(motion_file)
        assert self.motion_files, f"No motion files found for: {motion_file}"
        self._body_indexes = body_indexes
        self.joint_names = list(joint_names)
        self._legacy_joint_names = list(legacy_joint_names)
        self._load_motions(device)
        self.time_step_total = self.joint_pos.shape[0]
        self._build_episode_boundary_tensors(device)

    def _resolve_motion_files(self, motion_file: str) -> list[str]:
        motion_path = os.path.abspath(os.path.expanduser(motion_file))
        if os.path.isdir(motion_path):
            return sorted(
                os.path.join(root, filename)
                for root, _, filenames in os.walk(motion_path)
                for filename in sorted(filenames)
                if filename.endswith(".npz")
            )
        assert os.path.isfile(motion_path), f"Invalid file path: {motion_file}"
        if motion_path.endswith(".json"):
            with open(motion_path, encoding="utf-8") as f:
                manifest = json.load(f)
            motions = manifest["motions"] if isinstance(manifest, dict) and "motions" in manifest else manifest
            files = []
            for motion in motions:
                if isinstance(motion, str):
                    files.append(motion)
                elif isinstance(motion, dict):
                    files.append(motion.get("output_path") or motion.get("motion_file") or motion.get("input_file"))
                else:
                    raise TypeError(f"Unsupported motion manifest entry: {motion!r}")
            return [os.path.abspath(os.path.expanduser(path)) for path in files if path]
        return [motion_path]

    def _load_motions(self, device: str):
        required_keys = (
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        fps_list = []
        traj_lengths = []
        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        expected_shapes = None

        def decode_names(values) -> list[str]:
            names = []
            for value in np.asarray(values).reshape(-1).tolist():
                names.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
            return names

        def reorder_joint_columns(values: np.ndarray, source_names: list[str], motion_path: str) -> np.ndarray:
            if values.ndim != 2 or values.shape[1] != len(source_names):
                raise ValueError(
                    f"Joint data shape/name mismatch in {motion_path}: shape={values.shape}, names={len(source_names)}"
                )
            if len(source_names) != len(set(source_names)):
                raise ValueError(f"Duplicate joint names in {motion_path}: {source_names}")
            missing = [name for name in self.joint_names if name not in source_names]
            extra = [name for name in source_names if name not in self.joint_names]
            if missing or extra:
                raise ValueError(f"Joint contract mismatch in {motion_path}: missing={missing}, extra={extra}")
            indices = [source_names.index(name) for name in self.joint_names]
            return values[:, indices]

        for motion_path in self.motion_files:
            assert os.path.isfile(motion_path), f"Invalid motion file: {motion_path}"
            with np.load(motion_path) as data:
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    raise KeyError(f"{motion_path} is missing motion keys: {missing_keys}")

                source_joint_names = (
                    decode_names(data["joint_names"]) if "joint_names" in data else self._legacy_joint_names
                )
                joint_pos = reorder_joint_columns(data["joint_pos"], source_joint_names, motion_path)
                joint_vel = reorder_joint_columns(data["joint_vel"], source_joint_names, motion_path)
                body_pos_w = data["body_pos_w"]
                body_quat_w = data["body_quat_w"]
                body_lin_vel_w = data["body_lin_vel_w"]
                body_ang_vel_w = data["body_ang_vel_w"]
                current_shapes = (
                    joint_pos.shape[1:],
                    joint_vel.shape[1:],
                    body_pos_w.shape[1:],
                    body_quat_w.shape[1:],
                    body_lin_vel_w.shape[1:],
                    body_ang_vel_w.shape[1:],
                )
                if expected_shapes is None:
                    expected_shapes = current_shapes
                elif current_shapes != expected_shapes:
                    raise ValueError(
                        f"Motion shape mismatch in {motion_path}: expected {expected_shapes}, got {current_shapes}"
                    )

                fps = data["fps"]
                fps_list.append(float(np.asarray(fps).reshape(-1)[0]))
                traj_lengths.append(int(joint_pos.shape[0]))
                joint_pos_list.append(torch.tensor(joint_pos, dtype=torch.float32))
                joint_vel_list.append(torch.tensor(joint_vel, dtype=torch.float32))
                body_pos_w_list.append(torch.tensor(body_pos_w, dtype=torch.float32))
                body_quat_w_list.append(torch.tensor(body_quat_w, dtype=torch.float32))
                body_lin_vel_w_list.append(torch.tensor(body_lin_vel_w, dtype=torch.float32))
                body_ang_vel_w_list.append(torch.tensor(body_ang_vel_w, dtype=torch.float32))

        self.fps = np.asarray(fps_list, dtype=np.float32)
        if not np.allclose(self.fps, self.fps[0]):
            raise ValueError(f"All motion files must use the same fps, got {self.fps.tolist()}")
        self.traj_lengths = torch.tensor(traj_lengths, dtype=torch.long, device=device)
        self.joint_pos = torch.cat(joint_pos_list, dim=0).to(device)
        self.joint_vel = torch.cat(joint_vel_list, dim=0).to(device)
        self._body_pos_w = torch.cat(body_pos_w_list, dim=0).to(device)
        self._body_quat_w = torch.cat(body_quat_w_list, dim=0).to(device)
        self._body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0).to(device)
        self._body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0).to(device)

    def _build_episode_boundary_tensors(self, device: str):
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                torch.cumsum(self.traj_lengths, dim=0)[:-1],
            ],
            dim=0,
        )
        self.motion_offsets = offsets
        end_steps = offsets + self.traj_lengths
        valid_start_steps = []
        for offset, length in zip(offsets.tolist(), self.traj_lengths.tolist()):
            if length > 1:
                valid_start_steps.append(torch.arange(offset, offset + length - 1, dtype=torch.long, device=device))
        self.valid_start_steps = torch.cat(valid_start_steps, dim=0) if valid_start_steps else offsets
        self.end_markers = torch.zeros(self.time_step_total + 1, dtype=torch.bool, device=device)
        self.end_markers[end_steps] = True

    @property
    def has_multiple_motions(self) -> bool:
        return len(self.motion_files) > 1

    def sample_time_steps(self, count: int, start_ratio: float = 0.0) -> torch.Tensor:
        motion_ids = torch.randint(0, len(self.motion_files), (count,), device=self.traj_lengths.device)
        max_local_steps = torch.clamp(self.traj_lengths[motion_ids] - 1, min=1)
        local_steps = (torch.rand(count, device=self.traj_lengths.device) * max_local_steps).long()
        if start_ratio > 0.0:
            start_mask = torch.rand(count, device=self.traj_lengths.device) < start_ratio
            local_steps[start_mask] = 0
        return self.motion_offsets[motion_ids] + local_steps

    def is_done(self, time_steps: torch.Tensor) -> torch.Tensor:
        clamped = torch.clamp(time_steps, 0, self.time_step_total)
        return self.end_markers[clamped]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.motion_joint_names = list(self.cfg.motion_joint_names or self.robot.data.joint_names)
        self.joint_indexes = torch.tensor(
            self.robot.find_joints(self.motion_joint_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        self.motion = MotionLoader(
            self.cfg.motion_file,
            self.body_indexes,
            self.motion_joint_names,
            self.robot.data.joint_names,
            device=self.device,
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.joint_indexes]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self.joint_indexes]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if self.motion.has_multiple_motions:
            self.time_steps[env_ids] = self.motion.sample_time_steps(
                len(env_ids), start_ratio=self.cfg.motion_start_reset_ratio
            )
            self.metrics["sampling_entropy"][:] = 1.0
            self.metrics["sampling_top1_prob"][:] = 1.0 / max(len(self.motion.motion_files), 1)
            self.metrics["sampling_top1_bin"][:] = 0.0
            return

        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        if self.cfg.motion_start_reset_ratio > 0.0:
            start_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.motion_start_reset_ratio
            self.time_steps[env_ids[start_mask]] = 0

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.joint_indexes]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], joint_ids=self.joint_indexes, env_ids=env_ids
        )
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.motion.is_done(self.time_steps))[0]
        self._resample_command(env_ids)

        self.refresh_relative_body_targets()

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def refresh_relative_body_targets(self):
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    motion_joint_names: list[str] | None = None
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    motion_start_reset_ratio: float = 0.0

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
