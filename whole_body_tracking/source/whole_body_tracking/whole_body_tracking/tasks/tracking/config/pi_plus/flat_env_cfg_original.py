from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

# 引入小派模型
from whole_body_tracking.robots.pi_plus import PI_PLUS_21DOF_CFG
# 假设你已经把 agents 文件夹复制过来了，直接导入其中的低频缩放因子
from whole_body_tracking.tasks.tracking.config.pi_plus.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class PiPlusFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 1. 替换机器人资产
        self.scene.robot = PI_PLUS_21DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 小派的动作缩放比例 (如果没有特殊定义，默认全局为 1.0)
        self.actions.joint_pos.scale = {".*": 1.0}
        
        # 2. 挂载我们生成的动作数据集
        self.commands.motion.dataset_path = "/data2/yangky/test/whole_body_tracking/artifacts/pi_plus_motion:v0/motion.npz"
        
        # 3. 根节点与全身追踪关键点映射 (完全适配小派的连杆名称)
        self.commands.motion.anchor_body_name = "waist_link"
        self.commands.motion.body_names = [
            "base_link",               # 对应 pelvis
            "l_hip_roll_link",         # 对应 left_hip
            "l_calf_link",             # 对应 left_knee
            "l_ankle_roll_link",       # 对应 left_foot
            "r_hip_roll_link",         # 对应 right_hip
            "r_calf_link",             # 对应 right_knee
            "r_ankle_roll_link",       # 对应 right_foot
            "waist_link",              # 对应 torso/spine
            "l_shoulder_roll_link",    # 对应 left_shoulder
            "l_elbow_link",            # 对应 left_elbow
            "l_wrist_link",            # 对应 left_wrist
            "r_shoulder_roll_link",    # 对应 right_shoulder
            "r_elbow_link",            # 对应 right_elbow
            "r_wrist_link",            # 对应 right_wrist
        ]

        # 4. === 修复基类中硬编码的 G1 设定 ===
        # 修复质心随机化
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_link")
        
        # 修复防撞惩罚 (只允许手、脚碰地)
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!l_ankle_roll_link$)(?!r_ankle_roll_link$)(?!l_wrist_link$)(?!r_wrist_link$).+$"
            ],
        )
        
        # 修复末端位置误差终止条件
        self.terminations.ee_body_pos.params["body_names"] = [
            "l_ankle_roll_link",
            "r_ankle_roll_link",
            "l_wrist_link",
            "r_wrist_link",
        ]


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
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE