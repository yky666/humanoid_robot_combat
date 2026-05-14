import os
import pickle

import imageio
import mujoco
import numpy as np


# =========================
# 配置区
# =========================
PKL_PATH = "outputs/new_grab_clean_g1_motion.pkl"
XML_PATH = "assets/unitree_g1/g1_mocap_29dof.xml"
OUTPUT_VIDEO = "outputs/new_grab_clean_g1_render_fixed.mp4"

WIDTH = 960
HEIGHT = 720

# G1 站立时 pelvis/root 的大致高度
TARGET_MIN_ROOT_Z = 0.78


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """
    pkl 中 root_rot 是 xyzw；
    MuJoCo free joint qpos[3:7] 需要 wxyz。
    """
    return q_xyzw[[3, 0, 1, 2]]


def main():
    with open(PKL_PATH, "rb") as f:
        motion = pickle.load(f)

    root_pos = motion["root_pos"].copy()
    root_rot_xyzw = motion["root_rot"].copy()
    dof_pos = motion["dof_pos"].copy()
    fps = int(motion["fps"])

    num_frames = len(dof_pos)

    print(f"[INFO] Frames: {num_frames}, FPS: {fps}")
    print(f"[INFO] root_pos z range before fix: {root_pos[:, 2].min():.4f} ~ {root_pos[:, 2].max():.4f}")

    # =========================
    # 关键修正 1：抬高 root 高度
    # =========================
    z_offset = TARGET_MIN_ROOT_Z - root_pos[:, 2].min()
    root_pos[:, 2] += z_offset

    print(f"[INFO] z_offset: {z_offset:.4f}")
    print(f"[INFO] root_pos z range after fix: {root_pos[:, 2].min():.4f} ~ {root_pos[:, 2].max():.4f}")

    # =========================
    # 加载 MuJoCo 模型
    # =========================
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data_mj = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT)

    frames = []

    for i in range(num_frames):
        qpos = np.zeros(model.nq)

        # root position
        qpos[0:3] = root_pos[i]

        # =========================
        # 关键修正 2：xyzw -> wxyz
        # =========================
        qpos[3:7] = xyzw_to_wxyz(root_rot_xyzw[i])

        # 29 dof joints
        qpos[7:] = dof_pos[i]

        data_mj.qpos[:] = qpos
        mujoco.mj_forward(model, data_mj)

        renderer.update_scene(data_mj)
        frame = renderer.render()
        frames.append(frame)

        if i % 30 == 0:
            print(f"[INFO] Rendering {i}/{num_frames}")

    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    imageio.mimsave(OUTPUT_VIDEO, frames, fps=fps)

    print(f"\n✅ Fixed video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()