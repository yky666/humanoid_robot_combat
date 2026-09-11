"""Official SDK walk policy as an action prior for FG72 baoquan training.

Isaac residual actions are relative to the baoquan PD offset. The official
walk net is residual to the official default pose, and its 72-D frame scales
joint velocity by 0.05. Feeding Isaac policy observations into the MNN net
directly is therefore wrong.

This wrapper reconstructs the official observation from simulator state,
runs ``251111_180036_saw_50k.mnn``, converts teacher joint targets into
baoquan-residual actions, mixes them into the lower body, and holds the
arms at the baoquan pose (action 0).
"""

from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from whole_body_tracking.robots.t800 import T800_ACTION_SCALE
from whole_body_tracking.robots.t800_joint_order import T800_POLICY_JOINT_NAMES
from whole_body_tracking.tasks.fixed_guard.guard_contract import T800_BAOQUAN_POLICY_POSE

SDK_JOINT_NAMES = (
    T800_POLICY_JOINT_NAMES[:12]
    + T800_POLICY_JOINT_NAMES[13:18]
    + T800_POLICY_JOINT_NAMES[18:23]
)

OFFICIAL_DEFAULT_Q = [
    -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,
    -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,
    0.0, 0.15, 0.0, -0.25, 0.0,
    0.0, -0.15, 0.0, -0.25, 0.0,
]

DEFAULT_MNN = (
    "/mnt/data/yangky/test/engineai_robotics_native_sdk/assets/config/"
    "t800/rl_walking_example/policies/251111_180036_saw_50k.mnn"
)

OBS_CLIP = 100.0
VEL_SCALE = 0.05
COMMAND_SCALE = (2.0, 2.0, 1.0)
HISTORY = 15
FRAME = 72
ACTIONS = 22


class MnnWalkTeacher:
    """Batched CPU inference for the official walk MNN graph."""

    def __init__(self, model_path: str):
        import MNN

        self._MNN = MNN
        self.net = MNN.Interpreter(model_path)
        self.session = self.net.createSession({})
        self._batch: int | None = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        n = int(obs.shape[0])
        if self._batch != n:
            inp = self.net.getSessionInput(self.session)
            self.net.resizeTensor(inp, (n, FRAME * HISTORY + 3))
            self.net.resizeSession(self.session)
            self._batch = n
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        inp = self.net.getSessionInput(self.session)
        out = self.net.getSessionOutput(self.session)
        host_in = self._MNN.Tensor(
            [n, FRAME * HISTORY + 3],
            self._MNN.Halide_Type_Float,
            obs,
            self._MNN.Tensor_DimensionType_Caffe,
        )
        inp.copyFrom(host_in)
        self.net.runSession(self.session)
        host_out = self._MNN.Tensor(
            out.getShape(),
            self._MNN.Halide_Type_Float,
            np.zeros((n, ACTIONS), dtype=np.float32),
            self._MNN.Tensor_DimensionType_Caffe,
        )
        out.copyToHostTensor(host_out)
        return np.array(host_out.getData(), dtype=np.float32).reshape(n, ACTIONS)


class OfficialWalkPriorWrapper(gym.Wrapper):
    """Mix official walk actions into FG72 baoquan-residual actions."""

    def __init__(
        self,
        env,
        *,
        model_path: str | None = None,
        mix_start: float = 0.80,
        mix_end: float = 0.30,
        anneal_steps: int = 200_000,
        arm_keep: float = 1.0,
        imit_weight: float = 0.08,
    ):
        super().__init__(env)
        path = model_path or os.environ.get("FG72_WALK_PRIOR_MNN", DEFAULT_MNN)
        if not Path(path).is_file():
            raise FileNotFoundError(f"official walk MNN not found: {path}")
        self.teacher = MnnWalkTeacher(path)
        self.mix_start = float(os.environ.get("FG72_WALK_PRIOR_MIX", mix_start))
        self.mix_end = float(os.environ.get("FG72_WALK_PRIOR_MIX_END", mix_end))
        self.anneal_steps = int(os.environ.get("FG72_WALK_PRIOR_ANNEAL_STEPS", anneal_steps))
        self.arm_keep = float(os.environ.get("FG72_WALK_PRIOR_ARM_KEEP", arm_keep))
        self.imit_weight = float(os.environ.get("FG72_WALK_PRIOR_IMIT", imit_weight))
        self._calls = 0
        self._joint_ids = None
        self._hist = None
        self._last_official = None
        self._fill = None
        self._ready = False
        print(
            "[walk-prior] official MNN "
            f"mix={self.mix_start}->{self.mix_end} anneal={self.anneal_steps} "
            f"arm_keep={self.arm_keep} path={path}"
        )

    def _tensors(self):
        base = self.env.unwrapped
        n = int(base.num_envs)
        device = base.device
        if self._hist is None or self._hist.shape[0] != n:
            self._hist = torch.zeros(n, HISTORY, FRAME, device=device)
            self._last_official = torch.zeros(n, ACTIONS, device=device)
            self._fill = torch.ones(n, dtype=torch.bool, device=device)
        if self._joint_ids is None:
            robot = base.scene["robot"]
            ids, names = robot.find_joints(list(SDK_JOINT_NAMES), preserve_order=True)
            if len(ids) != ACTIONS:
                raise RuntimeError(f"expected {ACTIONS} SDK joints, got {names}")
            self._joint_ids = torch.as_tensor(ids, device=device, dtype=torch.long)
            offset_map = dict(getattr(base.cfg.actions.joint_pos, "offset", {}) or {})
            q_pd = [
                float(offset_map.get(name, T800_BAOQUAN_POLICY_POSE[name]))
                for name in SDK_JOINT_NAMES
            ]
            scale = [float(T800_ACTION_SCALE[name]) for name in SDK_JOINT_NAMES]
            self._q_off = torch.tensor(OFFICIAL_DEFAULT_Q, device=device, dtype=torch.float32)
            self._q_pd = torch.tensor(q_pd, device=device, dtype=torch.float32)
            self._scale = torch.tensor(scale, device=device, dtype=torch.float32)
            self._cmd_scale = torch.tensor(COMMAND_SCALE, device=device, dtype=torch.float32)
            # Overlay official gait on the current PD pose (slight crouch).
            # Arms still map official residual onto baoquan PD.
            self._delta = (self._q_off - self._q_pd) / self._scale
            self._delta[:12] = 0.0
            self._leg_crouch = self._q_pd[:12] - self._q_off[:12]
        return base

    def _mix_ratio(self) -> float:
        if self.anneal_steps <= 0:
            return self.mix_start
        t = min(1.0, self._calls / float(self.anneal_steps))
        return self.mix_end + (self.mix_start - self.mix_end) * (1.0 - t)

    def _official_frame(self, base) -> torch.Tensor:
        robot = base.scene["robot"]
        q = robot.data.joint_pos.index_select(1, self._joint_ids).clone()
        qd = robot.data.joint_vel.index_select(1, self._joint_ids).clone()
        # Un-crouch the legs so the official net sees its training pose, then
        # hide baoquan arms.
        q[:, :12] = q[:, :12] - self._leg_crouch
        q[:, 12:] = self._q_off[12:]
        qd[:, 12:] = 0.0
        ang = robot.data.root_ang_vel_b
        grav = robot.data.projected_gravity_b
        pos = q - self._q_off
        vel = qd * VEL_SCALE
        frame = torch.cat((pos, vel, self._last_official, ang, grav), dim=-1)
        return frame.clamp(-OBS_CLIP, OBS_CLIP)

    def _push_history(self, frame: torch.Tensor) -> torch.Tensor:
        fill = self._fill
        keep = ~fill
        if torch.any(keep):
            self._hist[keep, :-1] = self._hist[keep, 1:].clone()
            self._hist[keep, -1] = frame[keep]
        if torch.any(fill):
            self._hist[fill] = frame[fill].unsqueeze(1).expand(-1, HISTORY, -1)
            self._fill[fill] = False
        return self._hist.reshape(frame.shape[0], HISTORY * FRAME)

    def _teacher_student_action(self, base, student: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frame = self._official_frame(base)
        hist = self._push_history(frame)
        command = base.command_manager.get_command("base_velocity")[:, :3] * self._cmd_scale
        obs = torch.cat((hist, command), dim=-1)
        teacher_np = self.teacher.act(obs.detach().to("cpu").numpy())
        teacher_off = torch.as_tensor(teacher_np, device=student.device, dtype=student.dtype)
        teacher_off = teacher_off.clamp(-8.0, 8.0)
        teacher_stu = teacher_off + self._delta
        return teacher_off, teacher_stu

    def _blend(self, student: torch.Tensor, teacher_stu: torch.Tensor, mix: float) -> torch.Tensor:
        mixed = student.clone()
        mixed[:, :12] = mix * teacher_stu[:, :12] + (1.0 - mix) * student[:, :12]
        mixed[:, 12:] = (1.0 - self.arm_keep) * student[:, 12:]
        return mixed

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._tensors()
        self._fill[:] = True
        self._last_official.zero_()
        return obs, info

    def get_observations(self):
        if hasattr(self.env, "get_observations"):
            return self.env.get_observations()
        return self.env.unwrapped.observation_manager.compute()

    def step(self, action):
        base = self._tensors()
        student = action if torch.is_tensor(action) else torch.as_tensor(action, device=base.device)
        student = student.to(device=base.device, dtype=torch.float32)
        teacher_off, teacher_stu = self._teacher_student_action(base, student)
        mix = self._mix_ratio()
        mixed = self._blend(student, teacher_stu, mix)
        self._last_official = mixed - self._delta
        self._last_official[:, 12:] = 0.0
        obs, reward, terminated, truncated, info = self.env.step(mixed)
        if self.imit_weight:
            err = (student[:, :12] - teacher_stu[:, :12]).pow(2).mean(dim=-1)
            reward = reward + self.imit_weight * torch.exp(-err / 0.25)
        dones = terminated | truncated
        if torch.any(dones):
            self._fill[dones] = True
            self._last_official[dones] = 0.0
        self._calls += 1
        if self._calls <= 3 or self._calls % 50 == 0:
            robot = base.scene["robot"]
            pos = robot.data.root_pos_w[0].detach()
            vel = robot.data.root_lin_vel_w[0].detach()
            print(
                f"[walk-prior] t={self._calls} mix={mix:.2f} "
                f"xyz=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                f"vxy=({vel[0]:.2f},{vel[1]:.2f}) "
                f"a_off={teacher_off[0, :6].detach().cpu().numpy().round(3).tolist()}",
                flush=True,
            )
        if hasattr(base, "extras") and isinstance(base.extras, dict):
            base.extras["walk_prior_mix"] = torch.as_tensor(mix, device=base.device)
            base.extras["walk_prior_imit"] = (student[:, :12] - teacher_stu[:, :12]).abs().mean()
        return obs, reward, terminated, truncated, info


def maybe_wrap_official_walk_prior(env):
    """Enable with FG72_WALK_PRIOR=1. No-op otherwise."""
    flag = os.environ.get("FG72_WALK_PRIOR", "0").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return env
    return OfficialWalkPriorWrapper(env)
