import mujoco
import mujoco.viewer
import numpy as np
import pickle
import imageio
import os

# ========= 配置 =========
PKL_PATH = "outputs/g1_motion.pkl"
XML_PATH = "assets/unitree_g1/g1_mocap_29dof.xml"
OUTPUT_VIDEO = "outputs/g1_render.mp4"

# ========= 加载数据 =========
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

root_pos = data["root_pos"]
root_rot = data["root_rot"]  # xyzw
dof_pos = data["dof_pos"]
fps = data["fps"]

num_frames = len(dof_pos)

print(f"[INFO] Frames: {num_frames}, FPS: {fps}")

# ========= 加载 MuJoCo =========
model = mujoco.MjModel.from_xml_path(XML_PATH)
data_mj = mujoco.MjData(model)

# ========= 渲染器（关键：离屏）=========
renderer = mujoco.Renderer(model, width=640, height=480)

frames = []

# ========= 主循环 =========
for i in range(num_frames):
    qpos = np.zeros(model.nq)

    # root
    qpos[0:3] = root_pos[i]
    qpos[3:7] = root_rot[i]

    # dof
    qpos[7:] = dof_pos[i]

    data_mj.qpos[:] = qpos
    mujoco.mj_forward(model, data_mj)

    renderer.update_scene(data_mj)
    frame = renderer.render()
    frames.append(frame)

    if i % 30 == 0:
        print(f"[INFO] Rendering {i}/{num_frames}")

# ========= 保存视频 =========
os.makedirs("outputs", exist_ok=True)
imageio.mimsave(OUTPUT_VIDEO, frames, fps=fps)

print(f"\n✅ 视频已生成: {OUTPUT_VIDEO}")