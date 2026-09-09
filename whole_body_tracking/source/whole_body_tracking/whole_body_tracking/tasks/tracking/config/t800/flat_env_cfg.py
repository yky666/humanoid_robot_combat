from pathlib import Path

import whole_body_tracking.tasks.tracking.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from . import t800_mdp
from whole_body_tracking.robots.t800 import T800_ACTION_SCALE, T800_CFG
from whole_body_tracking.robots.t800_joint_order import T800_POLICY_JOINT_NAMES
from whole_body_tracking.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


def _resolve_t800_motion_file() -> str:
    """Locate a default T800 motion file for quick bring-up."""
    candidates = []
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        candidates.append(parent / "artifacts" / "boxing_T800" / "victory_50hz.npz")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    # Fallback to the expected relative layout under the repository root.
    return str(candidates[0])


@configclass
class T800FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0
        self.scene.robot = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos = t800_mdp.ResidualRefJointPositionActionCfg(
            asset_name="robot",
            joint_names=T800_POLICY_JOINT_NAMES,
            command_name="motion",
            preserve_order=True,
        )
        self.actions.joint_pos.scale = T800_ACTION_SCALE
        policy_joint_asset_cfg = SceneEntityCfg(
            "robot", joint_names=T800_POLICY_JOINT_NAMES, preserve_order=True
        )
        self.observations.policy.joint_pos.params = {"asset_cfg": policy_joint_asset_cfg}
        self.observations.policy.joint_vel.params = {"asset_cfg": policy_joint_asset_cfg}
        self.observations.critic.joint_pos.params = {"asset_cfg": policy_joint_asset_cfg}
        self.observations.critic.joint_vel.params = {"asset_cfg": policy_joint_asset_cfg}
        self.commands.motion.anchor_body_name = "LINK_BASE"
        self.commands.motion.motion_file = _resolve_t800_motion_file()
        self.commands.motion.motion_joint_names = T800_POLICY_JOINT_NAMES
        self.commands.motion.motion_body_names = t800_mdp.T800_MOTION_BODY_NAMES
        self.commands.motion.motion_start_reset_ratio = 0.35
        self.commands.motion.pose_range = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.005, 0.005),
            "roll": (-0.06, 0.06),
            "pitch": (-0.06, 0.06),
            "yaw": (-0.12, 0.12),
        }
        self.commands.motion.joint_position_range = (-0.05, 0.05)
        self.commands.motion.body_names = [
            "LINK_BASE",
            "LINK_HIP_ROLL_L",
            "LINK_KNEE_PITCH_L",
            "LINK_ANKLE_ROLL_L",
            "LINK_HIP_ROLL_R",
            "LINK_KNEE_PITCH_R",
            "LINK_ANKLE_ROLL_R",
            "LINK_TORSO_YAW",
            "LINK_SHOULDER_PITCH_L",
            "LINK_ELBOW_PITCH_L",
            "LINK_ELBOW_YAW_L",
            "LINK_SHOULDER_PITCH_R",
            "LINK_ELBOW_PITCH_R",
            "LINK_ELBOW_YAW_R",
            "LINK_HEAD_PITCH",
            "LINK_HEAD_YAW",
        ]
        self.events.base_com.params["asset_cfg"].body_names = "LINK_BASE"
        self.events.push_robot = None
        self.rewards.action_rate_l2.weight = -2e-2
        self.rewards.joint_limit.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!LINK_ANKLE_ROLL_L$)(?!LINK_ANKLE_ROLL_R$)(?!LINK_ELBOW_YAW_L$)(?!LINK_ELBOW_YAW_R$).+$"
        ]
        self.rewards.motion_global_anchor_pos.weight = 0.5
        self.rewards.motion_global_anchor_ori.weight = 0.5
        self.rewards.motion_body_pos.weight = 1.5
        self.rewards.motion_body_ori.weight = 1.25
        self.rewards.undesired_contacts.weight = -0.1
        self.terminations.ee_body_pos.params["body_names"] = [
            "LINK_ANKLE_ROLL_L",
            "LINK_ANKLE_ROLL_R",
            "LINK_ELBOW_YAW_L",
            "LINK_ELBOW_YAW_R",
        ]
        self.terminations.anchor_pos.params["threshold"] = 0.35
        self.terminations.ee_body_pos.params["threshold"] = 0.4


@configclass
class T800RecoveryEnvCfg(T800FlatEnvCfg):
    """Reference-tracking task for contact-rich prone and supine recovery."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 12.0
        # Keep half of resets at the exact competition preparation pose while
        # adaptive sampling continues to train difficult intermediate phases.
        self.commands.motion.motion_start_reset_ratio = 0.5
        self.commands.motion.pose_range = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.003, 0.003),
            "roll": (-0.04, 0.04),
            "pitch": (-0.04, 0.04),
            "yaw": (-0.08, 0.08),
        }
        self.commands.motion.joint_position_range = (-0.03, 0.03)

        # Recovery initially needs a controlled contact distribution. Broader
        # material randomization and external pushes belong in a later phase.
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.0)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.1)
        self.events.push_robot = None

        # Back, torso, arm, knee, and foot contact can all be intentional during
        # a get-up. Head contact remains strongly discouraged.
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "LINK_HEAD_PITCH",
            "LINK_HEAD_YAW",
        ]
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.joint_limit.weight = -5.0
        self.rewards.action_rate_l2.weight = -1e-2
        self.terminations.anchor_pos.params["threshold"] = 0.45
        self.terminations.ee_body_pos.params["threshold"] = 0.55


@configclass
class T800DirectGetupEnvCfg(T800FlatEnvCfg):
    """Reference-free T800 get-up exploration task.

    This is a simulation research task. Its observation contract is different
    from the 140-element whole-body-tracking policy used by the current Native
    SDK runner, so a trained policy still needs a dedicated deployment runner.
    """

    orientation: str = "mixed"
    target_height: float = 0.72
    target_joint_pos: list[float] = t800_mdp.T800_APPROX_BOXING_READY

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 8.0
        self.scene.robot = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        for joint_name, joint_pos in zip(T800_POLICY_JOINT_NAMES, self.target_joint_pos, strict=True):
            self.scene.robot.init_state.joint_pos[joint_name] = joint_pos

        self.commands.motion = None
        policy_joint_asset_cfg = SceneEntityCfg(
            "robot", joint_names=T800_POLICY_JOINT_NAMES, preserve_order=True
        )
        direct_action_scale = {
            joint_name: max(1.0, float(T800_ACTION_SCALE[joint_name]) * 5.0)
            for joint_name in T800_POLICY_JOINT_NAMES
        }
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=T800_POLICY_JOINT_NAMES,
            scale=direct_action_scale,
            use_default_offset=True,
            preserve_order=True,
        )

        target_params = {
            "asset_cfg": policy_joint_asset_cfg,
            "target_joint_pos": self.target_joint_pos,
        }
        height_params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": self.target_height,
        }
        self.observations.policy.command = ObsTerm(
            func=t800_mdp.getup_target_joint_error,
            params=target_params,
        )
        self.observations.policy.motion_anchor_pos_b = ObsTerm(
            func=t800_mdp.getup_root_height_error,
            params=height_params,
        )
        self.observations.policy.motion_anchor_ori_b = ObsTerm(func=mdp.projected_gravity)
        self.observations.policy.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        self.observations.policy.base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        self.observations.policy.joint_pos.params = {"asset_cfg": policy_joint_asset_cfg}
        self.observations.policy.joint_vel.params = {"asset_cfg": policy_joint_asset_cfg}

        self.observations.critic.command = ObsTerm(
            func=t800_mdp.getup_target_joint_error,
            params=target_params,
        )
        self.observations.critic.motion_anchor_pos_b = ObsTerm(
            func=t800_mdp.getup_root_height_error,
            params=height_params,
        )
        self.observations.critic.motion_anchor_ori_b = ObsTerm(func=mdp.projected_gravity)
        self.observations.critic.body_pos = ObsTerm(func=mdp.root_pos_w)
        self.observations.critic.body_ori = ObsTerm(func=mdp.projected_gravity)
        self.observations.critic.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        self.observations.critic.base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        self.observations.critic.joint_pos.params = {"asset_cfg": policy_joint_asset_cfg}
        self.observations.critic.joint_vel.params = {"asset_cfg": policy_joint_asset_cfg}

        self.events.add_joint_default_pos = None
        self.events.base_com.params["asset_cfg"].body_names = "LINK_BASE"
        self.events.push_robot = None
        self.events.reset_getup_pose = EventTerm(
            func=t800_mdp.reset_t800_getup_pose,
            mode="reset",
            params={
                "asset_cfg": policy_joint_asset_cfg,
                "orientation": self.orientation,
                "root_height": 0.25,
                "pose_noise": {
                    "x": (-0.03, 0.03),
                    "y": (-0.03, 0.03),
                    "z": (-0.01, 0.01),
                    "roll": (-0.08, 0.08),
                    "pitch": (-0.08, 0.08),
                    "yaw": (-0.25, 0.25),
                },
                "joint_position_noise": (-0.04, 0.04),
                "velocity_noise": (-0.05, 0.05),
            },
        )

        self.rewards.motion_global_anchor_pos = RewTerm(
            func=t800_mdp.getup_root_height_exp,
            weight=2.0,
            params={**height_params, "std": 0.22},
        )
        self.rewards.motion_global_anchor_ori = RewTerm(
            func=t800_mdp.getup_upright_exp,
            weight=2.5,
            params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.7},
        )
        self.rewards.motion_body_pos = RewTerm(
            func=t800_mdp.getup_target_joint_pose_exp,
            weight=1.0,
            params={**target_params, "std": 0.55},
        )
        self.rewards.motion_body_ori = RewTerm(
            func=t800_mdp.getup_success_bonus,
            weight=8.0,
            params={
                **height_params,
                "target_joint_pos": self.target_joint_pos,
                "max_tilt_rad": 0.35,
                "max_height_error": 0.16,
                "max_joint_error": 0.45,
            },
        )
        self.rewards.motion_body_lin_vel = RewTerm(
            func=t800_mdp.getup_low_root_velocity_exp,
            weight=0.5,
            params={"asset_cfg": SceneEntityCfg("robot"), "std": 2.0},
        )
        self.rewards.motion_body_ang_vel = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_limit.weight = -5.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "LINK_HEAD_PITCH",
            "LINK_HEAD_YAW",
        ]
        self.rewards.undesired_contacts.weight = -2.0

        self.terminations.anchor_pos = DoneTerm(
            func=t800_mdp.getup_root_xy_out_of_bounds,
            params={"asset_cfg": SceneEntityCfg("robot"), "max_distance": 3.5},
        )
        self.terminations.anchor_ori = DoneTerm(
            func=t800_mdp.getup_head_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["LINK_HEAD_PITCH", "LINK_HEAD_YAW"],
                ),
                "threshold": 400.0,
            },
        )
        self.terminations.ee_body_pos = None


@configclass
class T800DirectGetupProneEnvCfg(T800DirectGetupEnvCfg):
    orientation: str = "prone"


@configclass
class T800DirectGetupSupineEnvCfg(T800DirectGetupEnvCfg):
    orientation: str = "supine"


@configclass
class T800DirectGetupShapedEnvCfg(T800DirectGetupEnvCfg):
    """Direct get-up task with broader dense shaping for cold-start RL."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0
        self.events.reset_getup_pose.params["pose_noise"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.005, 0.005),
            "roll": (-0.04, 0.04),
            "pitch": (-0.04, 0.04),
            "yaw": (-0.12, 0.12),
        }
        self.events.reset_getup_pose.params["joint_position_noise"] = (-0.025, 0.025)
        self.events.reset_getup_pose.params["velocity_noise"] = (-0.025, 0.025)

        self.rewards.motion_global_anchor_pos.weight = 1.5
        self.rewards.motion_global_anchor_pos.params["std"] = 0.5
        self.rewards.motion_global_anchor_ori.weight = 1.5
        self.rewards.motion_global_anchor_ori.params["std"] = 1.8
        self.rewards.motion_body_pos.weight = 0.5
        self.rewards.motion_body_pos.params["std"] = 1.2
        self.rewards.motion_body_ori.weight = 10.0
        self.rewards.motion_body_lin_vel.weight = 0.25
        self.rewards.motion_body_ang_vel.weight = -0.025
        self.rewards.undesired_contacts.weight = -0.5
        self.terminations.anchor_ori = None

        self.rewards.getup_upright_progress = RewTerm(
            func=t800_mdp.getup_upright_linear,
            weight=3.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.getup_height_progress = RewTerm(
            func=t800_mdp.getup_root_height_linear,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_height": self.target_height,
                "max_error": 0.75,
            },
        )


@configclass
class T800DirectGetupProneShapedEnvCfg(T800DirectGetupShapedEnvCfg):
    orientation: str = "prone"


@configclass
class T800DirectGetupSupineShapedEnvCfg(T800DirectGetupShapedEnvCfg):
    orientation: str = "supine"


@configclass
class T800DirectGetupStagedEnvCfg(T800DirectGetupShapedEnvCfg):
    """Direct get-up task with staged height, stability, and actuator-margin shaping."""

    def __post_init__(self):
        super().__post_init__()

        policy_joint_asset_cfg = SceneEntityCfg(
            "robot", joint_names=T800_POLICY_JOINT_NAMES, preserve_order=True
        )
        robot_asset_cfg = SceneEntityCfg("robot")

        self.rewards.getup_height_stages = RewTerm(
            func=t800_mdp.getup_root_height_stage_reward,
            weight=5.0,
            params={
                "asset_cfg": robot_asset_cfg,
                "thresholds": [0.35, 0.45, 0.55, 0.65, 0.72],
                "temperature": 0.035,
            },
        )
        self.rewards.getup_stability = RewTerm(
            func=t800_mdp.getup_stability_exp,
            weight=2.0,
            params={
                "asset_cfg": robot_asset_cfg,
                "min_height": 0.50,
                "velocity_std": 1.0,
            },
        )
        self.rewards.getup_guard_stability = RewTerm(
            func=t800_mdp.getup_guard_stability_exp,
            weight=4.0,
            params={
                "asset_cfg": policy_joint_asset_cfg,
                "target_height": self.target_height,
                "target_joint_pos": self.target_joint_pos,
                "max_tilt_rad": 0.45,
                "max_height_error": 0.18,
                "max_joint_error": 0.65,
                "velocity_std": 0.8,
                "joint_velocity_std": 2.0,
            },
        )
        self.rewards.joint_margin = RewTerm(
            func=t800_mdp.joint_soft_limit_margin_violation,
            weight=-1.0,
            params={
                "asset_cfg": policy_joint_asset_cfg,
                "margin": 0.10,
            },
        )
        self.rewards.joint_velocity_margin = RewTerm(
            func=t800_mdp.joint_velocity_limit_violation,
            weight=-0.5,
            params={
                "asset_cfg": policy_joint_asset_cfg,
                "max_fraction": 0.80,
            },
        )
        self.rewards.joint_torque_margin = RewTerm(
            func=t800_mdp.joint_torque_limit_violation,
            weight=-0.25,
            params={
                "asset_cfg": policy_joint_asset_cfg,
                "max_fraction": 0.80,
            },
        )

        self.rewards.motion_body_ori.params["max_tilt_rad"] = 0.30
        self.rewards.motion_body_ori.params["max_height_error"] = 0.12
        self.rewards.motion_body_ori.params["max_joint_error"] = 0.35


@configclass
class T800DirectGetupProneStagedEnvCfg(T800DirectGetupStagedEnvCfg):
    orientation: str = "prone"


@configclass
class T800DirectGetupSupineStagedEnvCfg(T800DirectGetupStagedEnvCfg):
    orientation: str = "supine"


@configclass
class T800DirectGetupCurriculumEnvCfg(T800DirectGetupStagedEnvCfg):
    """Staged get-up task with reachable high-pose shaping before the strict gate."""

    def __post_init__(self):
        super().__post_init__()

        policy_joint_asset_cfg = SceneEntityCfg(
            "robot", joint_names=T800_POLICY_JOINT_NAMES, preserve_order=True
        )
        robot_asset_cfg = SceneEntityCfg("robot")

        self.rewards.motion_global_anchor_ori.weight = 2.0
        self.rewards.motion_body_pos.weight = 0.8
        self.rewards.motion_body_pos.params["std"] = 1.1
        self.rewards.motion_body_ori.weight = 12.0
        self.rewards.motion_body_lin_vel.weight = 0.4
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.getup_stability.weight = 3.0
        self.rewards.getup_stability.params["min_height"] = 0.55
        self.rewards.getup_guard_stability.weight = 2.0
        self.rewards.getup_guard_stability.params["max_tilt_rad"] = 0.80
        self.rewards.getup_guard_stability.params["max_height_error"] = 0.25
        self.rewards.getup_guard_stability.params["max_joint_error"] = 1.20
        self.rewards.getup_guard_stability.params["velocity_std"] = 1.0
        self.rewards.getup_guard_stability.params["joint_velocity_std"] = 3.0

        self.rewards.getup_high_upright = RewTerm(
            func=t800_mdp.getup_height_gated_upright_exp,
            weight=4.0,
            params={
                "asset_cfg": robot_asset_cfg,
                "min_height": 0.58,
                "std": 1.1,
                "height_temperature": 0.06,
            },
        )
        self.rewards.getup_high_joint_pose = RewTerm(
            func=t800_mdp.getup_height_gated_joint_pose_exp,
            weight=2.5,
            params={
                "asset_cfg": policy_joint_asset_cfg,
                "min_height": 0.58,
                "target_joint_pos": self.target_joint_pos,
                "std": 1.4,
                "height_temperature": 0.06,
            },
        )
        self.rewards.getup_high_low_velocity = RewTerm(
            func=t800_mdp.getup_height_gated_low_velocity_exp,
            weight=2.5,
            params={
                "asset_cfg": policy_joint_asset_cfg,
                "min_height": 0.58,
                "velocity_std": 1.2,
                "joint_velocity_std": 3.0,
                "height_temperature": 0.06,
            },
        )


@configclass
class T800DirectGetupProneCurriculumEnvCfg(T800DirectGetupCurriculumEnvCfg):
    orientation: str = "prone"


@configclass
class T800DirectGetupSupineCurriculumEnvCfg(T800DirectGetupCurriculumEnvCfg):
    orientation: str = "supine"


@configclass
class T800FlatWoStateEstimationEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class T800FlatLowFreqEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
