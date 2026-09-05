from pathlib import Path

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

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
