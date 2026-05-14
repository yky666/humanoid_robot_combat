from typing import Tuple, List, Literal

def load_robot_cfg(robot_type: str)->Tuple[dict, dict]:

    if robot_type == "g1_29d" or robot_type == "g1":
        from robotlib.beyondMimic.robots.g1 import G1_OPENSOURCE_CFG
        from robotlib.trackerLab.align_cfg import G129DOF_MOTION_ALIGN_CFG_GMR
        return G1_OPENSOURCE_CFG, G129DOF_MOTION_ALIGN_CFG_GMR
    if robot_type == "r2" or robot_type == "r2b":
        from robotlib.trackerLab.assets.humanoids.r2 import R2_CFG
        from robotlib.trackerLab.align_cfg import R2B_MOTION_ALIGN_CFG_GMR
        return R2_CFG, R2B_MOTION_ALIGN_CFG_GMR

    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")
    
def get_asset_config(robot_dir):
    return
