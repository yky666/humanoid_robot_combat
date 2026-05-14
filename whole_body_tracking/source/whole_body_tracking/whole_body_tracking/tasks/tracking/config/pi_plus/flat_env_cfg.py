from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp

# 引入小派模型
from whole_body_tracking.robots.pi_plus import PI_PLUS_21DOF_CFG
from whole_body_tracking.tasks.tracking.config.pi_plus.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class PiPlusFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 1. 替换机器人资产
        self.scene.robot = PI_PLUS_21DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = {".*": 1.0}
        
        # 2. 挂载我们生成的动作数据集 (射门动作专属路径)
        self.commands.motion.dataset_path = "/data2/yangky/test/whole_body_tracking/artifacts/pi_plus_soccer_kick:v0/motion.npz"

        # 学长之前已有的动作数据集(好像xml不太一样)
        # self.commands.motion.dataset_path = "/data2/yangky/test/datasets/soccer/足球策略文件汇总/football_data/pi_plus_goal.npz"
        
        # 3. 根节点与全身追踪关键点映射
        self.commands.motion.anchor_body_name = "waist_link"
        self.commands.motion.body_names = [
            "base_link", "l_hip_roll_link", "l_calf_link", "l_ankle_roll_link",
            "r_hip_roll_link", "r_calf_link", "r_ankle_roll_link", "waist_link",
            "l_shoulder_roll_link", "l_elbow_link", "l_wrist_link",
            "r_shoulder_roll_link", "r_elbow_link", "r_wrist_link",
        ]

        # 4. 修复质心随机化
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_link")
        
        # =====================================================================
        # 🎯 [踢球/行走专属]: 极其严格的接触惩罚
        # =====================================================================
        # 踢球动作中，除了两只脚底板，身体的任何其他部位(包括手、膝盖、屁股)碰地，
        # 都会被判定为严重失误并扣除巨额分数。
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[r"^(?!l_ankle_roll_link$)(?!r_ankle_roll_link$).+$"],
        )
        
        # 修复末端位置误差终止条件
        self.terminations.ee_body_pos.params["body_names"] = [
            "l_ankle_roll_link", "r_ankle_roll_link", "l_wrist_link", "r_wrist_link"
        ]

        # =====================================================================
        # 🛡️ [稳定性工程]: 消除高频震颤，逼迫学习真实物理平衡
        # =====================================================================
        # 惩罚动作突变：如果两帧之间发出的指令差值太大，就扣分 (消除帕金森抖动)
        if hasattr(self.rewards, "action_rate_l2"):
            self.rewards.action_rate_l2.weight = -0.05
        
        # 惩罚过大的力矩：防止机器人用蛮力去强行贴合轨迹，促使动作变得柔和且省电
        if hasattr(self.rewards, "dof_torques_l2"):
            self.rewards.dof_torques_l2.weight = -2e-5

        # =====================================================================
        # 🌪️ [域随机化]: 拒绝温室花朵，在恶劣环境中学会对抗
        # =====================================================================
        # 随机推力：每 2~3 秒，随机给机器人躯干一个来自四面八方的冲量，逼迫它岔开腿稳住底盘
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.0, 3.0),
            params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "z": (0.0, 0.0)}},
        )
        
        # 摩擦力随机化：让它不要养成“滑步”的坏习惯，在各种冰面和柏油路上都能站稳
        # 这一部分先暂时删掉，后面sim2real部分再进行fine tune
        # self.events.friction_randomization = EventTerm(
        #     func=mdp.randomize_rigid_body_material,
        #     mode="reset",
        #     params={
        #         "asset_cfg": SceneEntityCfg("ground"),
        #         "static_friction_range": (0.3, 1.2),
        #         "dynamic_friction_range": (0.3, 1.2),
        #         "restitution_range": (0.0, 0.0),  # 👈 新增：强制要求的弹性参数，设为0表示不反弹
        #         "num_buckets": 64,
        #     },
        # )


@configclass
class PiPlusFlatWoStateEstimationEnvCfg(PiPlusFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class PiPlusFlatLowFreqEnvCfg(PiPlusFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        if hasattr(self.rewards, "action_rate_l2"):
            self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE