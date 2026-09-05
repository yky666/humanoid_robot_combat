import os
import sys

# 1. 这里的路径请根据你第一步输出的结果进行修改
ISAACSIM_DIR = "/data2/yangky/miniconda/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim"

# 2. 强行将相关的扩展目录加入系统路径
exts_path = os.path.join(ISAACSIM_DIR, "exts")
if os.path.exists(exts_path):
    sys.path.append(os.path.join(exts_path, "omni.isaac.core"))
    sys.path.append(os.path.join(exts_path, "omni.isaac.lab")) # 如果有这个目录的话

# 3. 设置环境变量，确保 AppLauncher 能找到它
os.environ["ISAACSIM_PATH"] = ISAACSIM_DIR

# --- 原有的代码从这里开始 ---
from omni.isaac.lab.app import AppLauncher
# ... 其余代码

# 1. 配置启动器 (必须在导入其他 omni 库之前)
parser = argparse.ArgumentParser(description="创建一个简单的 Isaac Lab 场景")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 导入场景相关组件
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.sim import SimulationCfg, SimulationContext

def main():
    # 配置仿真参数
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    
    # 设置地面
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.0])
    prim_utils.create_prim("/World/GroundPlane", "VisualPlane", translation=(0, 0, 0))
    
    # 创建一个简单的立方体
    prim_utils.create_prim(
        "/World/Cube", 
        "Cube", 
        translation=(0, 0, 0.5), 
        scale=(0.5, 0.5, 0.5),
        attributes={"visualMaterial": ""}
    )

    # 重置仿真
    sim.reset()
    print("[INFO]: 场景创建完毕，正在运行...")

    # 仿真主循环
    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()