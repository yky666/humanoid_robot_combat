# 带地面的空场景（推荐使用）
from isaaclab_sim import Simulation, SimulationCfg
from isaaclab_assets.visuals import GroundPlane, GroundPlaneCfg

# 初始化仿真
sim_cfg = SimulationCfg()
sim = Simulation(sim_cfg)

# 添加地面
ground = GroundPlane(GroundPlaneCfg())

# 运行循环
while sim.is_running():
    sim.step()
