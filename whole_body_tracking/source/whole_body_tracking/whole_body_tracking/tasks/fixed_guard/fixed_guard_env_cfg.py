"""Online velocity-locomotion configuration for the T800 guard gait."""

from __future__ import annotations

import math

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import randomize_rigid_body_material
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_tracking.robots.t800 import T800_ACTION_SCALE, T800_CFG
from whole_body_tracking.tasks.tracking.config.t800.t800_mdp import T800_DFS_JOINT_NAMES
from whole_body_tracking.tasks.tracking.tracking_env_cfg import MySceneCfg

from . import curriculum
from . import rewards as guard_rewards
from .actions import FixedGuardJointPositionActionCfg
from .commands import FixedGuardVelocityCommandCfg
from .events import reset_guard_joints
from isaaclab.envs.mdp import rewards as isaac_rewards

from .guard_contract import (
    T800_BAOQUAN_HEAD_POSITION,
    T800_BAOQUAN_LEG_POSITION,
    T800_BAOQUAN_POLICY_POSE,
    T800_BAOQUAN_TORSO_YAW,
    T800_STAND_ROOT_HEIGHT,
    T800_TRAINING_JOINT_NAMES,
    TRAINING_GUARD_JOINT_NAMES,
    load_guard_pose,
)

_GUARD_SDK_NAMES, GUARD_POSITION = load_guard_pose()
GUARD_JOINT_NAMES = TRAINING_GUARD_JOINT_NAMES
FOOT_BODIES = ["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]
NON_FOOT_BODIES = r"^(?!LINK_ANKLE_ROLL_L$)(?!LINK_ANKLE_ROLL_R$).+"
LOWER_BODY_JOINT_NAMES = T800_TRAINING_JOINT_NAMES[:12]
TORSO_HEAD_JOINT_NAMES = ["J12_TORSO_YAW", "J27_HEAD_PITCH", "J28_HEAD_YAW"]
TORSO_HEAD_POSITION = [T800_BAOQUAN_TORSO_YAW, *T800_BAOQUAN_HEAD_POSITION]


@configclass
class FixedGuardSceneCfg(MySceneCfg):
    """Flat scene reusing the validated T800 asset and contact sensor."""


@configclass
class FixedGuardCommandsCfg:
    base_velocity = FixedGuardVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=1.0,
        heading_command=False,
        debug_vis=False,
        ranges=velocity_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 0.8),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class FixedGuardActionsCfg:
    joint_pos = FixedGuardJointPositionActionCfg(
        asset_name="robot",
        joint_names=T800_DFS_JOINT_NAMES,
        preserve_order=True,
        scale=T800_ACTION_SCALE,
        use_default_offset=False,
        offset=dict(T800_BAOQUAN_POLICY_POSE),
        guard_joint_names=GUARD_JOINT_NAMES,
        guard_joint_position=GUARD_POSITION,
        fixed_joint_names=["J27_HEAD_PITCH", "J28_HEAD_YAW"],
        fixed_joint_position=[0.0, 0.0],
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
class FixedGuardObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=velocity_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=velocity_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=velocity_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=velocity_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=velocity_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=T800_DFS_JOINT_NAMES, preserve_order=True
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=velocity_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=T800_DFS_JOINT_NAMES, preserve_order=True
                )
            },
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=velocity_mdp.last_action)
        gait_phase = ObsTerm(func=guard_rewards.gait_phase_observation)
        feet_positions_b = ObsTerm(
            func=guard_rewards.feet_positions_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=FOOT_BODIES, preserve_order=True
                )
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FixedGuardEventsCfg:
    physics_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 16,
        },
    )
    reset_base = EventTerm(
        func=velocity_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
            },
        },
    )
    reset_joints = EventTerm(
        func=velocity_mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (0.0, 0.0)},
    )
    reset_guard = EventTerm(
        func=reset_guard_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=GUARD_JOINT_NAMES, preserve_order=True
            ),
            "guard_position": GUARD_POSITION,
        },
    )
    reset_torso_head = EventTerm(
        func=reset_guard_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=TORSO_HEAD_JOINT_NAMES, preserve_order=True
            ),
            "guard_position": TORSO_HEAD_POSITION,
        },
    )
    reset_baoquan_legs = EventTerm(
        func=reset_guard_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LOWER_BODY_JOINT_NAMES, preserve_order=True
            ),
            "guard_position": T800_BAOQUAN_LEG_POSITION,
        },
    )


@configclass
class FixedGuardRewardsCfg:
    track_lin_vel_xy = RewTerm(
        func=guard_rewards.track_lin_vel_xy_gait_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "gate_floor": 0.2,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=FOOT_BODIES, preserve_order=True
            ),
        },
    )
    track_ang_vel_z = RewTerm(
        func=guard_rewards.track_ang_vel_z_gait_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "gate_floor": 0.2,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=FOOT_BODIES, preserve_order=True
            ),
        },
    )
    command_lin_vel_error = RewTerm(
        func=guard_rewards.command_lin_vel_error_l2,
        weight=-0.5,
        params={"command_name": "base_velocity"},
    )
    command_ang_vel_z_error = RewTerm(
        func=guard_rewards.command_ang_vel_z_error_l2,
        weight=-2.0,
        params={"command_name": "base_velocity"},
    )
    upright = RewTerm(func=velocity_mdp.flat_orientation_l2, weight=-1.0)
    vertical_velocity = RewTerm(func=velocity_mdp.lin_vel_z_l2, weight=-1.0)
    horizontal_ang_velocity = RewTerm(func=velocity_mdp.ang_vel_xy_l2, weight=-0.05)
    feet_slide = RewTerm(
        func=guard_rewards.feet_slide_safe,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
        },
    )
    joint_limits = RewTerm(func=velocity_mdp.joint_pos_limits, weight=-2.0)
    action_rate = RewTerm(func=velocity_mdp.action_rate_l2, weight=-0.01)
    joint_torques = RewTerm(func=velocity_mdp.joint_torques_l2, weight=-1.0e-5)
    joint_acceleration = RewTerm(func=velocity_mdp.joint_acc_l2, weight=-2.5e-7)
    guard_pose = RewTerm(
        func=guard_rewards.guard_pose_l2,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=GUARD_JOINT_NAMES, preserve_order=True
            ),
            "target": GUARD_POSITION,
        },
    )
    guard_velocity = RewTerm(
        func=guard_rewards.guard_velocity_l2,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=GUARD_JOINT_NAMES, preserve_order=True
            )
        },
    )
    baoquan_leg_pose = RewTerm(
        func=guard_rewards.guard_pose_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=LOWER_BODY_JOINT_NAMES, preserve_order=True
            ),
            "target": T800_BAOQUAN_LEG_POSITION,
        },
    )
    is_alive = RewTerm(func=isaac_rewards.is_alive, weight=1.0)
    base_height = RewTerm(
        func=isaac_rewards.base_height_l2,
        weight=-4.0,
        params={"target_height": T800_STAND_ROOT_HEIGHT},
    )
    gait_swing_force = RewTerm(
        func=guard_rewards.gait_swing_force,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=FOOT_BODIES, preserve_order=True
            ),
            "force_scale": 400.0,
            "force_std": 0.08,
            "delta_t": 0.02,
        },
    )
    gait_stance_speed = RewTerm(
        func=guard_rewards.gait_stance_speed,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=FOOT_BODIES, preserve_order=True
            ),
            "speed_std": 0.15,
            "delta_t": 0.02,
        },
    )
    gait_stance_support_force = RewTerm(
        func=guard_rewards.gait_stance_support_force,
        weight=0.6,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=FOOT_BODIES, preserve_order=True
            ),
            "force_scale": 400.0,
            "force_std": 0.15,
            "delta_t": 0.02,
        },
    )
    gait_swing_foot_clearance = RewTerm(
        func=guard_rewards.gait_swing_foot_clearance,
        # T800 needs explicit clearance shaping in addition to TienKung's
        # force/contact timing; the command-aligned placement term below keeps
        # this from being satisfied by a high backward kick.
        weight=0.4,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=FOOT_BODIES, preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=FOOT_BODIES, preserve_order=True
            ),
            "target_height": 0.12,
            "std": 0.04,
            "delta_t": 0.02,
        },
    )
    gait_swing_forward_placement = RewTerm(
        func=guard_rewards.gait_swing_forward_placement,
        # Require the unloaded foot to move to the command-consistent side of
        # the stance foot; this prevents a backward kick from satisfying the
        # swing-force/clearance terms alone.
        weight=1.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=FOOT_BODIES, preserve_order=True
            ),
            "cycle_s": 0.85,
            "air_ratio": 0.38,
            "delta_t": 0.02,
            "min_step_length": 0.05,
            "max_step_length": 0.24,
            "std": 0.08,
        },
    )
    feet_too_near = RewTerm(
        func=guard_rewards.feet_too_near,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=FOOT_BODIES, preserve_order=True
            ),
            "threshold": 0.2,
        },
    )
    feet_stumble = RewTerm(
        func=guard_rewards.feet_stumble,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=FOOT_BODIES, preserve_order=True
            )
        },
    )
    feet_force = RewTerm(
        func=guard_rewards.feet_force,
        weight=-0.003,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=FOOT_BODIES, preserve_order=True
            ),
            "threshold": 500.0,
            "max_reward": 400.0,
        },
    )
    feet_y_distance = RewTerm(
        func=guard_rewards.feet_y_distance,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=FOOT_BODIES, preserve_order=True
            ),
            "target": 0.30,
        },
    )
    ankle_torque = RewTerm(
        func=guard_rewards.ankle_torque_l2,
        weight=-5.0e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "J04_ANKLE_PITCH_L",
                    "J05_ANKLE_ROLL_L",
                    "J10_ANKLE_PITCH_R",
                    "J11_ANKLE_ROLL_R",
                ],
                preserve_order=True,
            )
        },
    )
    ankle_action = RewTerm(
        func=guard_rewards.action_l1,
        weight=-1.0e-3,
        params={"action_indices": [4, 5, 10, 11]},
    )
    hip_roll_action = RewTerm(
        func=guard_rewards.action_l1,
        weight=-1.0e-2,
        params={"action_indices": [1, 7]},
    )
    hip_yaw_action = RewTerm(
        func=guard_rewards.action_l1,
        weight=-1.0e-2,
        params={"action_indices": [2, 8]},
    )
    # RewardManager multiplies weights by step_dt: -200 becomes -4 at dt=0.02.
    termination_penalty = RewTerm(func=velocity_mdp.is_terminated, weight=-200.0)


@configclass
class FixedGuardTerminationsCfg:
    time_out = DoneTerm(func=velocity_mdp.time_out, time_out=True)
    torso_contact = DoneTerm(
        func=guard_rewards.illegal_contact_safe,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LINK_BASE"]),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(
        func=velocity_mdp.bad_orientation,
        params={"limit_angle": 0.9, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class FixedGuardCurriculumCfg:
    command_range = CurrTerm(func=curriculum.velocity_command_curriculum)


@configclass
class T800FixedGuardVelocityEnvCfg(ManagerBasedRLEnvCfg):
    scene: FixedGuardSceneCfg = FixedGuardSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: FixedGuardObservationsCfg = FixedGuardObservationsCfg()
    actions: FixedGuardActionsCfg = FixedGuardActionsCfg()
    commands: FixedGuardCommandsCfg = FixedGuardCommandsCfg()
    rewards: FixedGuardRewardsCfg = FixedGuardRewardsCfg()
    terminations: FixedGuardTerminationsCfg = FixedGuardTerminationsCfg()
    events: FixedGuardEventsCfg = FixedGuardEventsCfg()
    curriculum: FixedGuardCurriculumCfg = FixedGuardCurriculumCfg()

    def __post_init__(self):
        self.scene.robot = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, T800_STAND_ROOT_HEIGHT)
        self.scene.robot.init_state.joint_pos = dict(T800_BAOQUAN_POLICY_POSE)
        # Crouch puts more load on the ankles than the official near-straight
        # default; keep the rest of T800_CFG and only stiffen the feet PD.
        self.scene.robot.actuators["feet"].stiffness = {
            ".*_ANKLE_PITCH.*": 80.0,
            ".*_ANKLE_ROLL.*": 80.0,
        }
        self.scene.robot.actuators["feet"].damping = {
            ".*_ANKLE_PITCH.*": 2.0,
            ".*_ANKLE_ROLL.*": 2.0,
        }
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Match TienKung's per-control-step force average: four 5 ms physics
        # samples are retained for each 20 ms policy step.
        self.scene.contact_forces.history_length = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


@configclass
class T800FixedGuardVelocityEnvCfg_PLAY(T800FixedGuardVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
