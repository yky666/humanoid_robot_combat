import argparse
import pathlib
import os
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print


if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gvhmr_pred_file",
        type=str,
        default="/data2/yangky/test/GVHMR/outputs/demo/grab/hmr4d_results.pt",
    )

    parser.add_argument(
        "--robot",
        default="unitree_g1",
    )

    parser.add_argument(
        "--save_path",
        default="outputs/g1_motion.pkl",   # ✅ 默认直接保存
    )

    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    # ===== 1. 加载人体动作 =====
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        args.gvhmr_pred_file, SMPLX_FOLDER
    )

    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    # ===== 2. 初始化 GMR =====
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    # ===== 3. 只做 retarget（不再用 viewer）=====
    print("[green]Start retargeting...[/green]")

    qpos_list = []

    for i in range(len(smplx_data_frames)):

        smplx_data = smplx_data_frames[i]

        # 核心：映射到机器人
        qpos = retarget.retarget(smplx_data)

        qpos_list.append(qpos)

        if i % 30 == 0:
            print(f"Processed frame {i}/{len(smplx_data_frames)}")

    print("[green]Retarget finished![/green]")

    # ===== 4. 保存结果 =====
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }

    import pickle
    with open(args.save_path, "wb") as f:
        pickle.dump(motion_data, f)

    print(f"[yellow]Saved to {args.save_path}[/yellow]")