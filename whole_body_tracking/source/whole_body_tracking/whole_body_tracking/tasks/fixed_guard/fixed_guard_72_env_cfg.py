"""SDK-compatible 72-D/22-D T800 guard-gait training configuration.

This task intentionally keeps the original SDK observation contract: one frame
contains 72 values and the three velocity-command values are appended outside
the 15-frame history by the training wrapper, matching the SDK runner.
"""

from __future__ import annotations

import os

import isaaclab.terrains as terrain_gen
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.envs.mdp import randomize_rigid_body_com, randomize_rigid_body_mass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_tracking.robots.t800 import T800_ACTION_SCALE
from whole_body_tracking.tasks.tracking.config.t800.t800_mdp import T800_DFS_JOINT_NAMES

from .actions import FixedGuardJointPositionActionCfg
from .guard_contract import (
    T800_BAOQUAN_LEG_POSITION,
    T800_BAOQUAN_POLICY_POSE,
    T800_TRAINING_JOINT_NAMES,
)
from .speed_bump import SpeedBumpStripsTerrainCfg
from .fixed_guard_env_cfg import (
    GUARD_JOINT_NAMES,
    GUARD_POSITION,
    FixedGuardActionsCfg,
    FixedGuardEventsCfg,
    T800FixedGuardVelocityEnvCfg,
)

# SDK controls the 12 lower-body joints and 10 arm joints. Torso and head are
# held by the surrounding SDK model and are deliberately excluded.
SDK_JOINT_NAMES = T800_DFS_JOINT_NAMES[:12] + T800_DFS_JOINT_NAMES[13:18] + T800_DFS_JOINT_NAMES[18:23]
# IsaacLab's JointPositionAction accepts a scalar or a name-indexed mapping,
# never a positional list.  The mapping retains the exact SDK joint order
# while allowing per-joint lower-body/arm action scales.
SDK_ACTION_SCALE = {name: T800_ACTION_SCALE[name] for name in SDK_JOINT_NAMES}

# <=10 cm rough terrain / discrete boxes for the robustness stage.
GUARD_LIGHT_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=4,
    num_cols=4,
    curriculum=False,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.40),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.40,
            noise_range=(0.01, 0.08),
            noise_step=0.01,
            border_width=0.25,
        ),
        "boxes": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.20, 0.55),
            obstacle_height_range=(0.03, 0.10),
            num_obstacles=10,
            platform_width=1.2,
        ),
    },
)


@configclass
class TransitionSafeCfg:
    """Parameters shared with the opt-in SDK transition-safe profile.

    They are configuration-only: the actor remains 72-D with the same 22-D
    action order.  Transition wrappers consume these values at reset and
    command activation boundaries instead of injecting simulator-only fields
    into policy observations.
    """

    duration_s: float = 0.5
    command_ramp_rate: tuple[float, float, float] = (1.0, 0.6, 1.5)
    max_target_delta: float = 0.03
    history_length: int = 15
    action_dim: int = 22
    action_delay_steps: tuple[int, int] = (0, 2)
    activation_time_s: tuple[float, float] = (0.0, 1.0)
    push_interval_s: tuple[float, float] = (10.0, 15.0)
    push_velocity_range: tuple[float, float] = (-0.25, 0.25)
    initial_linear_velocity: tuple[float, float] = (-0.05, 0.05)
    initial_angular_velocity: tuple[float, float] = (-0.10, 0.10)


@configclass
class FixedGuard72ActionsCfg(FixedGuardActionsCfg):
    """22-D action term in the same order as the SDK's legacy walk policy."""

    joint_pos = FixedGuardJointPositionActionCfg(
        asset_name="robot",
        joint_names=SDK_JOINT_NAMES,
        preserve_order=True,
        scale=SDK_ACTION_SCALE,
        use_default_offset=False,
        offset={name: T800_BAOQUAN_POLICY_POSE[name] for name in SDK_JOINT_NAMES},
        guard_joint_names=GUARD_JOINT_NAMES,
        guard_joint_position=GUARD_POSITION,
        fixed_joint_names=[],
        fixed_joint_position=[],
        compensation_joint_names=[
            "J14_SHOULDER_ROLL_L",
            "J15_SHOULDER_YAW_L",
            "J17_ELBOW_YAW_L",
            "J21_SHOULDER_ROLL_R",
            "J22_SHOULDER_YAW_R",
            "J24_ELBOW_YAW_R",
        ],
        compensation_limit=0.10,
        command_name="base_velocity",
        stop_command_threshold=0.1,
        stance_joint_names=[],
        stance_joint_position=[],
    )


@configclass
class FixedGuard72ObservationsCfg:
    """72-D SDK frame: no gait clock or foot-position extensions."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=velocity_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SDK_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=velocity_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SDK_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=velocity_mdp.last_action)
        # Keep the exact upstream SDK frame order: pos, vel, previous action,
        # body angular velocity, projected gravity.
        base_ang_vel = ObsTerm(
            func=velocity_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=velocity_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 15

    policy: PolicyCfg = PolicyCfg()


@configclass
class T800FixedGuardVelocity72EnvCfg(T800FixedGuardVelocityEnvCfg):
    """72-D/22-D training task; the original 97-D/25-D task is unchanged."""

    observations: FixedGuard72ObservationsCfg = FixedGuard72ObservationsCfg()
    actions: FixedGuard72ActionsCfg = FixedGuard72ActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Push linear/yaw tracking so the policy cannot hide in an in-place crouch.
        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_ang_vel_z.weight = 3.5
        self.rewards.track_lin_vel_xy.params["std"] = 0.35
        self.rewards.track_ang_vel_z.params["std"] = 0.35
        self.rewards.command_lin_vel_error.weight = -1.2
        self.rewards.command_ang_vel_z_error.weight = -2.5
        if os.environ.get("FG72_WALK_PRIOR", "0").strip().lower() in {"1", "true", "yes", "on"}:
            # Official walk stands/walks near the SDK default pose. Keeping a
            # strong baoquan-leg penalty fights that gait and recreates the
            # in-place crouch local minimum.
            self.rewards.baoquan_leg_pose.weight = 0.0
            self.rewards.base_height.weight = -2.0
            self.rewards.base_height.params["target_height"] = 1.00
            from whole_body_tracking.robots.t800 import T800_ACTION_OFFSET

            # Official gait overlaid on a light squat so COM sits ~5 cm lower.
            offset = {name: T800_BAOQUAN_POLICY_POSE[name] for name in SDK_JOINT_NAMES}
            spawn = dict(T800_BAOQUAN_POLICY_POSE)
            squat = {
                "J00_HIP_PITCH_L": -0.20,
                "J03_KNEE_PITCH_L": 0.34,
                "J04_ANKLE_PITCH_L": -0.14,
                "J06_HIP_PITCH_R": -0.20,
                "J09_KNEE_PITCH_R": 0.34,
                "J10_ANKLE_PITCH_R": -0.14,
            }
            for name in SDK_JOINT_NAMES[:12]:
                offset[name] = squat.get(name, T800_ACTION_OFFSET[name])
                spawn[name] = squat.get(name, T800_ACTION_OFFSET[name])
            self.actions.joint_pos.offset = offset
            self.scene.robot.init_state.joint_pos = spawn
            self.scene.robot.init_state.pos = (0.0, 0.0, 1.02)
            self.rewards.feet_y_distance.weight = -6.5
            self.rewards.feet_y_distance.params["target"] = 0.26
            self.rewards.feet_too_near.weight = -8.0
            self.rewards.feet_too_near.params["threshold"] = 0.20
            self.rewards.hip_roll_action.weight = -0.07
            self.rewards.hip_yaw_action.weight = -0.08
            self.rewards.feet_slide.weight = -2.0


@configclass
class T800FixedGuardVelocity72TransitionSafeEnvCfg(T800FixedGuardVelocity72EnvCfg):
    """Opt-in transition curriculum; legacy 72-D task remains unchanged."""

    transition_safe: TransitionSafeCfg = TransitionSafeCfg()

    @configclass
    class EventsCfg(FixedGuardEventsCfg):
        """TienKung-style startup and interval perturbations, opt-in only."""

        reset_base = EventTerm(
            func=velocity_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0),
                    "roll": (-0.10, 0.10), "pitch": (-0.10, 0.10), "yaw": (-0.10, 0.10),
                },
            },
        )
        pelvis_mass = EventTerm(
            func=randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["LINK_BASE"]),
                "mass_distribution_params": (0.95, 1.05),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        pelvis_com = EventTerm(
            func=randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["LINK_BASE"]),
                "com_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.015, 0.015)},
            },
        )
        push_robot = EventTerm(
            func=velocity_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "z": (0.0, 0.0),
                                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            },
        )

    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.20


@configclass
class T800FixedGuardVelocity72EnvCfg_PLAY(T800FixedGuardVelocity72EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        # Watchdog / play scripts set FG_PLAY_V* to emulate a joystick stick.
        self.curriculum.command_range = None
        vx = float(os.environ.get("FG_PLAY_VX", "0.0"))
        vy = float(os.environ.get("FG_PLAY_VY", "0.0"))
        wz = float(os.environ.get("FG_PLAY_WZ", "0.0"))
        self.commands.base_velocity.ranges.lin_vel_x = (vx, vx)
        self.commands.base_velocity.ranges.lin_vel_y = (vy, vy)
        self.commands.base_velocity.ranges.ang_vel_z = (wz, wz)
        self.commands.base_velocity.rel_standing_envs = 0.0
        if os.environ.get("FG_PLAY_WIDE", "0") == "1":
            self.scene.num_envs = 1
            self.viewer.eye = (6.0, -6.0, 3.0)
            self.viewer.lookat = (0.0, 0.0, 0.8)
            self.viewer.origin_type = "world"


GUARD_BUMP_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=1,
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=4,
    num_cols=4,
    curriculum=False,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),
        "speed_bump": SpeedBumpStripsTerrainCfg(
            proportion=0.50,
            bottom=0.35,
            top=0.10,
            height=0.07,
            num_strips=4,
            spawn_pad=1.5,
            spacing=1.7,
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.01, 0.04),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)


class T800FixedGuardVelocity72RoughEnvCfg(T800FixedGuardVelocity72EnvCfg):
    """Same 72-D/22-D policy on <=10 cm rough terrain and small boxes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = GUARD_LIGHT_TERRAIN_CFG
        self.scene.terrain.use_terrain_origins = True
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class T800FixedGuardVelocity72RoughEnvCfg_PLAY(T800FixedGuardVelocity72RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.curriculum.command_range = None
        vx = float(os.environ.get("FG_PLAY_VX", "0.3"))
        vy = float(os.environ.get("FG_PLAY_VY", "0.0"))
        wz = float(os.environ.get("FG_PLAY_WZ", "0.0"))
        self.commands.base_velocity.ranges.lin_vel_x = (vx, vx)
        self.commands.base_velocity.ranges.lin_vel_y = (vy, vy)
        self.commands.base_velocity.ranges.ang_vel_z = (wz, wz)
        self.commands.base_velocity.rel_standing_envs = 0.0
        if os.environ.get("FG_PLAY_WIDE", "0") == "1":
            self.scene.num_envs = 1
            self.viewer.eye = (6.0, -6.0, 3.0)
            self.viewer.lookat = (0.0, 0.0, 0.8)
            self.viewer.origin_type = "world"


@configclass
class T800FixedGuardVelocity72BumpEnvCfg(T800FixedGuardVelocity72EnvCfg):
    """Back/yaw locomotion over 70 mm trapezoidal speed bumps."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = GUARD_BUMP_TERRAIN_CFG
        self.scene.terrain.use_terrain_origins = True
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class T800FixedGuardVelocity72BumpEnvCfg_PLAY(T800FixedGuardVelocity72BumpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.curriculum.command_range = None
        vx = float(os.environ.get("FG_PLAY_VX", "-0.40"))
        vy = float(os.environ.get("FG_PLAY_VY", "0.0"))
        wz = float(os.environ.get("FG_PLAY_WZ", "0.0"))
        self.commands.base_velocity.ranges.lin_vel_x = (vx, vx)
        self.commands.base_velocity.ranges.lin_vel_y = (vy, vy)
        self.commands.base_velocity.ranges.ang_vel_z = (wz, wz)
        self.commands.base_velocity.rel_standing_envs = 0.0
        if os.environ.get("FG_PLAY_WIDE", "0") == "1":
            self.scene.num_envs = 1
            self.viewer.eye = (6.0, -6.0, 3.0)
            self.viewer.lookat = (0.0, 0.0, 0.8)
            self.viewer.origin_type = "world"
