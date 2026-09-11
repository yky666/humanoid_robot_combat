from __future__ import annotations

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as velocity_mdp

from whole_body_tracking.robots.t800 import T800_ACTION_SCALE, T800_CFG
from whole_body_tracking.robots.t800_joint_order import T800_POLICY_JOINT_NAMES
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from . import baoquan_mdp

LEG_JOINT_NAMES = T800_POLICY_JOINT_NAMES[:12]
UPPER_BODY_JOINT_NAMES = T800_POLICY_JOINT_NAMES[12:]
OFFICIAL_WALK_LEG_TARGET = [
    -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,
    -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,
]
BOXING_UPPER_TARGET = [
    -0.065857, -0.688721, 0.902954, -0.142029, -1.917786, -0.218201,
    -1.195740, -0.167419, 0.582397, -1.878113, 0.199158, 0.245972, 0.588745,
]
BOXING_READY_TARGET = OFFICIAL_WALK_LEG_TARGET + BOXING_UPPER_TARGET

LIGHT_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=0, size=(8.0, 8.0), border_width=2.0, num_rows=4, num_cols=4,
    curriculum=True, horizontal_scale=0.1, vertical_scale=0.005,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.50),
        "light_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.30, noise_range=(0.01, 0.035), noise_step=0.01, border_width=0.25),
        "light_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20, slope_range=(0.0, 0.15), platform_width=2.5, border_width=0.25),
    },
)

@configclass
class BaoquanCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(8.0, 12.0), rel_standing_envs=0.05,
        rel_heading_envs=0.0, heading_command=False, debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 0.8), lin_vel_y=(-0.8, 0.8), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)),
    )

@configclass
class BaoquanActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINT_NAMES,
        scale={name: float(T800_ACTION_SCALE[name]) * 0.25 for name in LEG_JOINT_NAMES},
        use_default_offset=True, preserve_order=True)

@configclass
class BaoquanObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.15, n_max=0.15))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.20, n_max=0.20))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)}, noise=Unoise(n_min=-1.0, n_max=1.0))
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()

@configclass
class BaoquanRewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-2.0)
    base_height = RewTerm(func=baoquan_mdp.base_height_l2_clipped, weight=-8.0, params={"target_height": 1.20, "asset_cfg": SceneEntityCfg("robot")})
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")})
    track_lin_vel_xy_exp = RewTerm(func=velocity_mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5, params={"command_name": "base_velocity", "std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=velocity_mdp.track_ang_vel_z_world_exp, weight=0.7, params={"command_name": "base_velocity", "std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.015)
    feet_air_time = RewTerm(func=velocity_mdp.feet_air_time_positive_biped, weight=0.20, params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]), "threshold": 0.35})
    feet_slide = RewTerm(func=velocity_mdp.feet_slide, weight=-0.15, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]), "asset_cfg": SceneEntityCfg("robot", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"])})
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-0.5, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[r"^(?!LINK_ANKLE_ROLL_L$)(?!LINK_ANKLE_ROLL_R$).+$"]), "threshold": 1.0})
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)})
    upper_guard = RewTerm(func=baoquan_mdp.upper_body_guard_error, weight=-0.8, params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINT_NAMES, preserve_order=True), "target_joint_pos": BOXING_UPPER_TARGET})
    upper_guard_velocity = RewTerm(func=baoquan_mdp.upper_body_guard_velocity, weight=-0.02, params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINT_NAMES, preserve_order=True)})

@configclass
class BaoquanTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=velocity_mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LINK_BASE"]), "threshold": 1.0})
    unstable_root = DoneTerm(func=baoquan_mdp.unstable_root, params={"target_height": 1.20, "max_height_error": 2.0, "max_linear_speed": 20.0})

@configclass
class BaoquanCurriculumCfg:
    terrain_levels = CurrTerm(func=velocity_mdp.terrain_levels_vel)

@configclass
class T800BaoquanLocomotionEnvCfg(TrackingEnvCfg):
    observations: BaoquanObservationsCfg = BaoquanObservationsCfg()
    actions: BaoquanActionsCfg = BaoquanActionsCfg()
    commands: BaoquanCommandsCfg = BaoquanCommandsCfg()
    rewards: BaoquanRewardsCfg = BaoquanRewardsCfg()
    terminations: BaoquanTerminationsCfg = BaoquanTerminationsCfg()
    curriculum: BaoquanCurriculumCfg = BaoquanCurriculumCfg()
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.robot = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 1.20)
        self.events.push_robot = None
        self.events.base_com = None
        self.events.add_joint_default_pos = None
        self.scene.contact_forces.update_period = self.sim.dt
        for joint_name, joint_pos in zip(T800_POLICY_JOINT_NAMES, BOXING_READY_TARGET, strict=True):
            self.scene.robot.init_state.joint_pos[joint_name] = joint_pos

@configclass
class T800BaoquanFlatWalkEnvCfg(T800BaoquanLocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-0.45, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.curriculum.terrain_levels = None

@configclass
class T800BaoquanFlatFullEnvCfg(T800BaoquanLocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels = None

@configclass
class T800BaoquanLightTerrainEnvCfg(T800BaoquanLocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = LIGHT_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.terrain.use_terrain_origins = True
