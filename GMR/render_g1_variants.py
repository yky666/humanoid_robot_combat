import os
import pickle
import math

import imageio
import mujoco
import numpy as np


PKL_PATH = "outputs/new_grab_clean_g1_motion.pkl"
XML_PATH = "assets/unitree_g1/g1_mocap_29dof.xml"
OUT_DIR = "outputs/orientation_test"

WIDTH = 960
HEIGHT = 720
TARGET_MIN_ROOT_Z = 0.78


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_mul_wxyz(q1, q2):
    """
    Hamilton product.
    q1, q2 are both wxyz.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def axis_angle_to_quat_wxyz(axis, angle_deg):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    angle = math.radians(angle_deg)
    half = angle / 2.0

    w = math.cos(half)
    xyz = axis * math.sin(half)

    return quat_normalize(np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64))


def xyzw_to_wxyz(q_xyzw):
    """
    pkl 里面 root_rot 是 xyzw；
    MuJoCo qpos[3:7] 需要 wxyz。
    """
    return quat_normalize(q_xyzw[[3, 0, 1, 2]])


def render_one_variant(name, correction_quat_wxyz):
    print(f"\n========== Rendering variant: {name} ==========")

    with open(PKL_PATH, "rb") as f:
        motion = pickle.load(f)

    root_pos = motion["root_pos"].copy()
    root_rot_xyzw = motion["root_rot"].copy()
    dof_pos = motion["dof_pos"].copy()
    fps = int(motion["fps"])

    num_frames = len(dof_pos)

    # 抬高 root，避免机器人贴地或穿地
    z_offset = TARGET_MIN_ROOT_Z - root_pos[:, 2].min()
    root_pos[:, 2] += z_offset

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data_mj = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT)

    frames = []

    for i in range(num_frames):
        qpos = np.zeros(model.nq)

        qpos[0:3] = root_pos[i]

        # 原始 root_rot: xyzw -> wxyz
        root_q_wxyz = xyzw_to_wxyz(root_rot_xyzw[i])

        # 关键：加全局旋转补偿
        # 这里采用 correction * root
        fixed_q = quat_mul_wxyz(correction_quat_wxyz, root_q_wxyz)
        fixed_q = quat_normalize(fixed_q)

        qpos[3:7] = fixed_q
        qpos[7:] = dof_pos[i]

        data_mj.qpos[:] = qpos
        mujoco.mj_forward(model, data_mj)

        renderer.update_scene(data_mj)
        frame = renderer.render()
        frames.append(frame)

        if i % 60 == 0:
            print(f"[{name}] Rendering {i}/{num_frames}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}.mp4")
    imageio.mimsave(out_path, frames, fps=fps)

    print(f"[OK] Saved: {out_path}")


def main():
    variants = {
        "identity": np.array([1.0, 0.0, 0.0, 0.0]),

        "x_plus_90": axis_angle_to_quat_wxyz([1, 0, 0], 90),
        "x_minus_90": axis_angle_to_quat_wxyz([1, 0, 0], -90),
        "x_180": axis_angle_to_quat_wxyz([1, 0, 0], 180),

        "y_plus_90": axis_angle_to_quat_wxyz([0, 1, 0], 90),
        "y_minus_90": axis_angle_to_quat_wxyz([0, 1, 0], -90),
        "y_180": axis_angle_to_quat_wxyz([0, 1, 0], 180),

        "z_plus_90": axis_angle_to_quat_wxyz([0, 0, 1], 90),
        "z_minus_90": axis_angle_to_quat_wxyz([0, 0, 1], -90),
        "z_180": axis_angle_to_quat_wxyz([0, 0, 1], 180),
    }

    for name, qcorr in variants.items():
        render_one_variant(name, qcorr)


if __name__ == "__main__":
    main()