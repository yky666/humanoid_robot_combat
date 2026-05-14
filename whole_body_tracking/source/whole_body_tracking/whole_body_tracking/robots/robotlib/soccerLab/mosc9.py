import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from robotlib import ROBOTLIB_ASSETLIB_DIR

joint_names = ['b_Lh', 'b_Rh', 'Lh_Ll', 'Rh_Rl', 'Ll_Ll1', 'Rl_Rl1', 'Ll1_Ll2', 'Rl1_Rl2', 'Ll2_La', 'Rl2_Ra', 'La_Lf', 'Ra_Rf']
body_names = [
    'body', 'Lhip', 'Llap', 'Lleg1', 'Lleg2', 
    'Lankle', 'Lfoot', 'Lshoulder', 'Larm', 
    'Lhand', 'Rhip', 'Rlap', 'Rleg1', 'Rleg2', 
    'Rankle', 'Rfoot', 'Rshoulder', 'Rarm', 
    'Rhand', 'neck', 'head'
]

MOSC9_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOTLIB_ASSETLIB_DIR}/third_party/MOSC/usd/MOSC_0516.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=45.0,
            damping=1.0,
            effort_limit=100,
        ),
    },
)