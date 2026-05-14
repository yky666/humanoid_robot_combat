import os
import pickle
import math

import imageio
import mujoco
import numpy as np


PKL_PATH = "outputs/new_task2_g1_motion.pkl"
XML_PATH = "assets/unitree_g1/g1_mocap_29dof.xml"
OUTPUT_VIDEO = "outputs/new_task2_g1_render_y_minus_90.mp4"

WIDTH = 960
HEIGHT = 720
TARGET_MIN_ROOT_Z = 0.78


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def xyzw_to_wxyz(q_xyzw):
    return quat_normalize(q_xyzw[[3, 0, 1, 2]])


def axis_angle_to_quat_wxyz(axis, angle_deg):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    angle = math.radians(angle_deg)
    half = angle / 2.0

    w = math.cos(half)
    xyz = axis * math.sin(half)

    return quat_normalize(np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64))


def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return quat_normalize(np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64))


def main():
    with open(PKL_PATH, "rb") as f:
        motion = pickle.load(f)

    root_pos = motion["root_pos"].copy()
    root_rot_xyzw = motion["root_rot"].copy()
    dof_pos = motion["dof_pos"].copy()
    fps = int(motion["fps"])

    print(f"[INFO] Frames: {len(dof_pos)}, FPS: {fps}")
    print(f"[INFO] root_pos z before: {root_pos[:, 2].min():.4f} ~ {root_pos[:, 2].max():.4f}")

    z_offset = TARGET_MIN_ROOT_Z - root_pos[:, 2].min()
    root_pos[:, 2] += z_offset

    print(f"[INFO] z_offset: {z_offset:.4f}")
    print(f"[INFO] root_pos z after:  {root_pos[:, 2].min():.4f} ~ {root_pos[:, 2].max():.4f}")

    correction_q = axis_angle_to_quat_wxyz([0, 1, 0], -90)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data_mj = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT)

    frames = []

    for i in range(len(dof_pos)):
        qpos = np.zeros(model.nq)

        qpos[0:3] = root_pos[i]

        root_q = xyzw_to_wxyz(root_rot_xyzw[i])
        fixed_q = quat_mul_wxyz(correction_q, root_q)

        qpos[3:7] = fixed_q
        qpos[7:] = dof_pos[i]

        data_mj.qpos[:] = qpos
        mujoco.mj_forward(model, data_mj)

        renderer.update_scene(data_mj)
        frame = renderer.render()
        frames.append(frame)

        if i % 30 == 0:
            print(f"[INFO] Rendering {i}/{len(dof_pos)}")

    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    imageio.mimsave(OUTPUT_VIDEO, frames, fps=fps)

    print(f"\n✅ Video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
