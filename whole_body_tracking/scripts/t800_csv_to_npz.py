"""This script replays a T800 motion and outputs a tracking-ready NPZ file.

.. code-block:: bash

    # Legacy CSV input: root_pos(xyz), root_quat(xyzw), dof_pos
    python t800_csv_to_npz.py --input_file /path/to/t800_motion.csv --input_fps 30 --output_name t800_motion --output_fps 50

    # GMR pickle input
    python t800_csv_to_npz.py --input_file /path/to/gmr_motion.npz --input_format gmr_pickle --output_name t800_motion

    # Portable GMR raw NPZ input: root_pos, root_rot(xyzw), dof_pos, fps
    python t800_csv_to_npz.py --input_file /path/to/gmr_motion_raw.npz --input_format gmr_npz --output_name t800_motion

    # Legacy 40-column numpy input:
    python t800_csv_to_npz.py --input_file /path/to/riot_combo.npy --input_format legacy_npy --output_name riot_combo_tracking
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import os
import pickle
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv/GMR pickle and output to npz file for t800.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file.")
parser.add_argument(
    "--input_format",
    type=str,
    choices=("auto", "csv", "gmr_pickle", "gmr_npz", "legacy_npy"),
    default="auto",
    help=(
        "Input motion format. Use gmr_pickle for GMR outputs saved via pickle, gmr_npz for portable raw GMR "
        "arrays, or legacy_npy for historic 40-column T800 arrays."
    ),
)
parser.add_argument(
    "--input_fps",
    type=int,
    default=None,
    help="The fps of the input motion. Required for csv. Optional for GMR pickle if the file already contains fps.",
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
parser.add_argument(
    "--output_path",
    type=str,
    default=None,
    help="Optional explicit output path. Defaults to /tmp/<output_name>.npz.",
)
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument(
    "--skip_wandb_upload",
    action="store_true",
    help="Skip uploading the generated NPZ as a wandb artifact.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs for t800
##
from whole_body_tracking.robots.t800 import T800_CFG
from whole_body_tracking.robots.t800_joint_order import (
    T800_JOINT_ORDER_VERSION,
    T800_POLICY_JOINT_NAMES,
)

T800_DFS_JOINT_NAMES = T800_POLICY_JOINT_NAMES


# Historic local T800 arrays store root pose followed by an interleaved 25-DoF joint block
# and 8 extra columns that are not consumed by the tracking loader.
LEGACY_NPY_JOINT_INDICES = [
    7,
    13,
    19,
    8,
    14,
    20,
    25,
    30,
    9,
    15,
    21,
    26,
    31,
    10,
    16,
    22,
    27,
    11,
    17,
    23,
    28,
    12,
    18,
    24,
    29,
]


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation (Replaced with t800)
    robot: ArticulationCfg = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = None
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from csv or GMR pickle output."""
        input_format = self._infer_input_format()
        if input_format == "csv":
            self._load_csv_motion()
        elif input_format == "gmr_pickle":
            self._load_gmr_pickle_motion()
        elif input_format == "gmr_npz":
            self._load_gmr_npz_motion()
        elif input_format == "legacy_npy":
            self._load_legacy_npy_motion()
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

        self.input_frames = self.motion_base_poss_input.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"Motion loaded ({self.motion_file}, format={input_format}), duration: {self.duration} sec, frames:"
            f" {self.input_frames}, input_fps: {self.input_fps}"
        )

    def _infer_input_format(self) -> str:
        if args_cli.input_format != "auto":
            return args_cli.input_format

        input_path = Path(self.motion_file)
        if input_path.suffix.lower() == ".csv":
            return "csv"
        if input_path.suffix.lower() == ".npy":
            arr = np.load(input_path, allow_pickle=False)
            if arr.ndim == 2 and arr.shape[1] == 40:
                return "legacy_npy"
        if input_path.suffix.lower() == ".npz":
            try:
                with np.load(input_path, allow_pickle=False) as motion:
                    if {"root_pos", "root_rot", "dof_pos"}.issubset(motion.files):
                        return "gmr_npz"
            except Exception:
                pass

        try:
            with open(input_path, "rb") as f:
                motion = pickle.load(f)
            if isinstance(motion, dict) and {"root_pos", "root_rot", "dof_pos"}.issubset(motion.keys()):
                return "gmr_pickle"
        except Exception:
            pass

        raise ValueError(
            f"Unable to infer input format for {self.motion_file}. Please pass --input_format csv, "
            "gmr_pickle, gmr_npz, or legacy_npy."
        )

    def _resolve_input_fps(self, file_fps: int | float | None = None) -> float:
        if self.input_fps is not None:
            return float(self.input_fps)
        if file_fps is not None:
            return float(file_fps)
        raise ValueError("Input FPS is required. Pass --input_fps or provide a GMR pickle containing an fps field.")

    def _apply_frame_range(self, *arrays: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.frame_range is None:
            return arrays
        start, end = self.frame_range
        frame_slice = slice(start - 1, end)
        return tuple(array[frame_slice] for array in arrays)

    def _load_csv_motion(self):
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        if motion.shape[1] != 7 + len(T800_DFS_JOINT_NAMES):
            raise ValueError(
                f"CSV motion should have {7 + len(T800_DFS_JOINT_NAMES)} columns "
                f"(xyz + xyzw + {len(T800_DFS_JOINT_NAMES)} dof), got {motion.shape[1]}"
            )

        self.input_fps = self._resolve_input_fps(file_fps=None)
        self.input_dt = 1.0 / self.input_fps
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        self.motion_dof_poss_input = motion[:, 7:]

    def _fps_from_gmr_value(self, value):
        if value is None:
            return None
        value = np.asarray(value)
        if value.size != 1:
            raise ValueError(f"GMR fps should be scalar, got shape {value.shape}")
        return float(value.reshape(-1)[0])

    def _load_gmr_arrays(self, motion):
        required_keys = {"root_pos", "root_rot", "dof_pos"}
        keys = set(motion.keys() if isinstance(motion, dict) else motion.files)
        missing_keys = required_keys.difference(keys)
        if missing_keys:
            raise KeyError(f"GMR motion is missing required keys: {sorted(missing_keys)}")

        file_fps = self._fps_from_gmr_value(motion["fps"]) if "fps" in keys else None
        self.input_fps = self._resolve_input_fps(file_fps=file_fps)
        self.input_dt = 1.0 / self.input_fps

        root_pos = torch.from_numpy(np.asarray(motion["root_pos"], dtype=np.float32)).to(self.device)
        root_rot_xyzw = torch.from_numpy(np.asarray(motion["root_rot"], dtype=np.float32)).to(self.device)
        dof_pos = torch.from_numpy(np.asarray(motion["dof_pos"], dtype=np.float32)).to(self.device)
        root_pos, root_rot_xyzw, dof_pos = self._apply_frame_range(root_pos, root_rot_xyzw, dof_pos)

        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"GMR root_pos should have shape (T, 3), got {tuple(root_pos.shape)}")
        if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
            raise ValueError(f"GMR root_rot should have shape (T, 4), got {tuple(root_rot_xyzw.shape)}")
        if dof_pos.ndim != 2 or dof_pos.shape[1] != len(T800_DFS_JOINT_NAMES):
            raise ValueError(
                f"GMR dof_pos should have shape (T, {len(T800_DFS_JOINT_NAMES)}), got {tuple(dof_pos.shape)}"
            )

        self.motion_base_poss_input = root_pos
        self.motion_base_rots_input = root_rot_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        self.motion_dof_poss_input = dof_pos

    def _load_gmr_pickle_motion(self):
        # Older or alternate numpy builds may serialize arrays under private module
        # paths such as "numpy._core". Map them to the local numpy install so the
        # GMR pickle can be loaded across environments.
        if "numpy._core" not in sys.modules:
            sys.modules["numpy._core"] = np.core
        if "numpy._core.numeric" not in sys.modules:
            sys.modules["numpy._core.numeric"] = np.core.numeric

        with open(self.motion_file, "rb") as f:
            motion = pickle.load(f)

        self._load_gmr_arrays(motion)

    def _load_gmr_npz_motion(self):
        with np.load(self.motion_file, allow_pickle=False) as motion:
            self._load_gmr_arrays(motion)

    def _load_legacy_npy_motion(self):
        motion = torch.from_numpy(np.load(self.motion_file, allow_pickle=False))
        motion = motion.to(torch.float32).to(self.device)
        if motion.ndim != 2 or motion.shape[1] != 40:
            raise ValueError(f"Legacy numpy motion should have shape (T, 40), got {tuple(motion.shape)}")

        if self.frame_range is not None:
            motion = self._apply_frame_range(motion)[0]

        self.input_fps = self._resolve_input_fps(file_fps=None)
        self.input_dt = 1.0 / self.input_fps
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, [6, 3, 4, 5]]  # xyzw -> wxyz
        self.motion_dof_poss_input = motion[:, LEGACY_NPY_JOINT_INDICES]

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        if self.input_frames < 2:
            raise ValueError("Need at least 2 frames to interpolate a motion.")
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations."""
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_names": np.asarray(joint_names),
        "joint_order_version": np.asarray(T800_JOINT_ORDER_VERSION),
        "body_names": np.asarray(robot.body_names),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, robot_joint_indexes].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, robot_joint_indexes].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            # Save locally
            output_path = args_cli.output_path or f"/tmp/{args_cli.output_name}.npz"
            output_path = str(Path(output_path).expanduser().resolve())
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savez(output_path, **log)
            print(f"[INFO]: Successfully saved NPZ to {output_path}")

            # Optionally sync to wandb
            if args_cli.skip_wandb_upload:
                print("[INFO]: Skipping wandb artifact upload.")
            else:
                try:
                    import wandb

                    collection = args_cli.output_name
                    run = wandb.init(project="csv_to_npz", name=collection)
                    print(f"[INFO]: Logging motion to wandb: {collection}")
                    registry = "motions"
                    logged_artifact = run.log_artifact(artifact_or_path=output_path, name=collection, type=registry)
                    run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{registry}/{collection}")
                    run.finish()
                    print(f"[INFO]: Motion saved to wandb registry: {registry}/{collection}")
                except ImportError:
                    print("[INFO]: wandb not installed, skipping remote logging.")
            
            # break after saving to stop headless run automatically
            break


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator with t800 joints exactly matching your config
    run_simulator(
        sim,
        scene,
        joint_names=T800_DFS_JOINT_NAMES,
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
