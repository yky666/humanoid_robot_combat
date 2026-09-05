# T800 动作重定向逻辑与映射规则

本文档总结当前 T800 入围赛动作资产的重定向实现，覆盖 GMR/SMPLX、LaFAN/BVH、MotionDecode-G1 以及进入 IsaacLab/BeyondMimic 训练前的格式转换。

## 1. 当前使用的主要路径

- GMR 根目录：`/data2/yangky/test/gmr`
- IsaacLab/BeyondMimic 训练目录：`/data2/yangky/test/whole_body_tracking`
- T800 GMR 目标模型：`gmr/assets/t800/t800.xml`
- T800 IsaacLab 目标模型：`whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/t800.py`
- T800 IK config：
  - `gmr/general_motion_retargeting/ik_configs/smplx_to_t800.json`
  - `gmr/general_motion_retargeting/ik_configs/bvh_lafan1_to_t800.json`
- 已人工审计通过的入围赛动作集合：
  - `whole_body_tracking/artifacts/t800_approved_qualifier_20260902`
  - `whole_body_tracking/configs/t800_qualifier_approved_20260902.json`

## 2. 总体数据流

### SMPLX/AMASS 到 T800

1. AMASS/ACCAD 等数据提供 SMPL-X 参数和人体 root/trans/body pose。
2. `gmr/scripts/smplx_to_robot_dataset.py` 调用 `GeneralMotionRetargeting(src_human="smplx", tgt_robot="t800")`。
3. GMR 每帧读取人体关键 body 的全局位置和姿态，按照 `smplx_to_t800.json` 构造 T800 body 的 IK 目标。
4. `mink.solve_ik` 在 T800 MuJoCo 模型上解出 `qpos`。
5. 保存 GMR pkl：`root_pos`, `root_rot`, `dof_pos`, `fps`, `local_body_pos`, `link_body_list`。
6. `whole_body_tracking/scripts/t800_csv_to_npz.py` 在 IsaacLab 中 replay 该运动，导出训练用 tracking NPZ。

### LaFAN/BVH 到 T800

LaFAN/BVH 的逻辑与 SMPLX 相同，只是源骨架名字不同，使用 `bvh_lafan1_to_t800.json`。目标侧仍然是同一套 T800 body 和 DoF。

### MotionDecode-G1 到 T800

MotionDecode samples 已经是 Unitree G1 root pose + 29 DoF CSV，不是原始人体参数。当前桥接脚本是 `gmr/scripts/motiondecode_g1_csv_to_t800.py`：

1. 保留 G1 CSV 中的 root trajectory。
2. 将 G1 里语义相同的腿、腰 yaw、肩、肘、腕 yaw 映射到 T800 的 25 DoF 顺序。
3. T800 没有 waist roll/pitch、wrist pitch/roll，因此这些源关节不导出。
4. T800 head pitch/yaw 在 MotionDecode 源里没有对应列，置 0。
5. 左右肘 pitch 做符号翻转，以匹配 T800 关节方向。
6. 默认按 T800 MJCF joint range 做 clip。
7. 使用 `--align-floor` 时，按 T800 collision geom 最低点计算一个固定 root-z offset，使最低采样点位于 `floor_height + floor_clearance`。

这条路径是 G1 轨迹到 T800 轨迹的语义关节映射，不是完整的 SMPLX/GMR IK。

## 3. GMR 的核心算法

实现入口：`gmr/general_motion_retargeting/motion_retarget.py`

关键步骤如下：

1. 加载目标机器人 MuJoCo XML，并枚举 `model.nv`, `model.nbody`, `model.nu`。
2. 根据 `src_human` 和 `tgt_robot` 从 `IK_CONFIG_DICT` 读取 IK config。
3. 如提供 `actual_human_height`，先计算
   `ratio = actual_human_height / human_height_assumption`，
   再将 `human_scale_table` 中每个 body 的 scale 乘上该 ratio。
4. 每帧把人体数据转为 numpy，先在人体 root 局部坐标中缩放位置，再转回全局。
5. 对每个匹配 body 应用 position offset 和 rotation offset。rotation offset 使用 scalar-first quaternion，也就是 `wxyz`。
6. 构造两批 `mink.FrameTask`：
   - `ik_match_table1`：第一阶段，通常更强调 root、脚、整体姿态。
   - `ik_match_table2`：第二阶段，补上 hip/knee/shoulder/elbow 等位置约束，并提高脚的方向权重。
7. 每一阶段最多迭代 10 次，若误差下降小于阈值则提前停止。
8. 返回当前 T800 `qpos`，其中：
   - `qpos[:3]` 是 root position。
   - `qpos[3:7]` 是 root quaternion，MuJoCo 内部为 `wxyz`。
   - `qpos[7:]` 是 T800 DoF。

GMR 是 body-frame IK，不是逐关节角直接复制。关节角由目标 body 的 SE(3) 约束和 T800 运动学共同决定。

## 4. T800 IK body 映射表

`smplx_to_t800.json` 和 `bvh_lafan1_to_t800.json` 的目标 body 完全一致。区别只是源 body 名称不同。

| T800 body | SMPLX source | LaFAN/BVH source | table1 pos/rot weight | table2 pos/rot weight | 说明 |
| --- | --- | --- | --- | --- | --- |
| `LINK_BASE` | `pelvis` | `Hips` | 100 / 10 | 100 / 5 | root 位置和姿态强约束 |
| `LINK_HIP_ROLL_L` | `left_hip` | `LeftUpLeg` | 0 / 10 | 10 / 5 | 左髋姿态先行，第二阶段补位置 |
| `LINK_KNEE_PITCH_L` | `left_knee` | `LeftLeg` | 0 / 10 | 10 / 5 | 左膝姿态先行，第二阶段补位置 |
| `LINK_ANKLE_ROLL_L` | `left_foot` | `LeftFootMod` | 100 / 10 | 100 / 50 | 左脚位置强约束，第二阶段脚姿态更强 |
| `LINK_HIP_ROLL_R` | `right_hip` | `RightUpLeg` | 0 / 10 | 10 / 5 | 右髋 |
| `LINK_KNEE_PITCH_R` | `right_knee` | `RightLeg` | 0 / 10 | 10 / 5 | 右膝 |
| `LINK_ANKLE_ROLL_R` | `right_foot` | `RightFootMod` | 100 / 10 | 100 / 50 | 右脚位置强约束，第二阶段脚姿态更强 |
| `LINK_TORSO_YAW` | `spine3` | `Spine3` | 0 / 10 | 0 / 10 | 躯干姿态约束 |
| `LINK_SHOULDER_PITCH_L` | `left_shoulder` | `LeftShoulder` | 0 / 10 | 10 / 5 | 左肩 |
| `LINK_ELBOW_PITCH_L` | `left_elbow` | `LeftArm` | 0 / 10 | 10 / 5 | 左肘 |
| `LINK_ELBOW_YAW_L` | `left_wrist` | `LeftForeArm` | 0 / 10 | 10 / 5 | 用腕/前臂方向约束 T800 肘 yaw |
| `LINK_SHOULDER_PITCH_R` | `right_shoulder` | `RightShoulder` | 0 / 10 | 10 / 5 | 右肩 |
| `LINK_ELBOW_PITCH_R` | `right_elbow` | `RightArm` | 0 / 10 | 10 / 5 | 右肘 |
| `LINK_ELBOW_YAW_R` | `right_wrist` | `RightForeArm` | 0 / 10 | 10 / 5 | 用腕/前臂方向约束 T800 肘 yaw |

### 位置 offset

- 大多数 body 的 position offset 为 `[0, 0, 0]`。
- 第一阶段左脚 `LINK_ANKLE_ROLL_L` 使用 `[0, 0.02, 0]`。
- 第一阶段右脚 `LINK_ANKLE_ROLL_R` 使用 `[0, -0.02, 0]`。
- 左右 `LINK_ELBOW_YAW_*` 使用 `[0, 0, -0.08]`，让腕/前臂目标更贴合 T800 肘 yaw link。

### 姿态 offset

IK config 中 quaternion 均为 scalar-first `wxyz`，在代码里通过 `R.from_quat(rot_offset, scalar_first=True)` 读取。

| 目标 body 类别 | rotation offset `wxyz` | 说明 |
| --- | --- | --- |
| base、knee、torso、第一阶段 ankle | `[0.5, -0.5, -0.5, -0.5]` | 人体坐标系到 T800 link frame 的基础旋转 |
| 第二阶段 ankle | `[-0.5, 0.5, 0.5, 0.5]` | 脚部第二阶段使用相反等价方向以强化足端姿态 |
| hip roll L/R | `[0.4267755, -0.5637931, -0.5637931, -0.4267755]` | 髋部 link frame 对齐 |
| left shoulder | `[0.70710678, 0, -0.70710678, 0]` | 左肩 frame 对齐 |
| left elbow / left elbow yaw | `[1, 0, 0, 0]` | 左肘局部 frame 基本一致 |
| right shoulder | `[0, 0.70710678, 0, 0.70710678]` | 右肩 frame 对齐 |
| right elbow / right elbow yaw | `[0, 0, 0, -1]` | 右肘 frame 对齐 |

## 5. 高度和地面对齐

GMR 本体提供了两个相关机制：

- `ground_height`：从 config 读入，目前 T800 config 为 `0.0`。
- `offset_human_data_to_ground()`：可按源人体最低 foot/Foot 点平移到地面，但当前 dataset 脚本主流程没有显式开启这个参数。

SMPLX dataset 转换脚本中额外做了目标机器人 FK 高度修正：

1. 用 T800 FK 计算每帧所有 body 的 world z。
2. 找到全片段最低点 `lowest_height`。
3. 执行 `root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset`。
4. 当前 `ground_offset = 0.0`，目标是让最低 body 恰好不低于地面。
5. 再执行 `root_pos[:, :2] -= root_pos[0, :2]`，把第一帧 xy 作为轨迹原点。

MotionDecode-G1 直接映射路径中，`--align-floor` 用 MuJoCo collision geom 的最低点做同类 root-z 修正，并可留 `floor_clearance`。

因此，“腿陷入地板”通常不是 IK 映射表本身的问题，而是某个路径缺少目标机器人 FK/collision 的 root-z 后处理，或者渲染时使用的模型与高度修正时的模型不一致。

## 6. T800 关节顺序契约

统一定义位于：
`whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/t800_joint_order.py`。

当前格式版本为 `t800_policy_v1`。必须区分以下三种顺序：

1. `T800_POLICY_JOINT_NAMES`：NPZ、训练 observation/action、ONNX 输入输出使用的 25 维语义顺序。
2. `T800_LEGACY_ISAAC_STORAGE_JOINT_NAMES`：IsaacLab 按 URDF 拓扑生成的 articulation 内部顺序；旧 NPZ 曾错误地直接保存该顺序。
3. `T800_SDK_POLICY_JOINT_NAMES`：EngineAI SDK 的电机命名顺序；语义顺序与 policy 一致，但右臂和头部的数字编号不同。

策略语义顺序如下：

顺序如下：

| index | T800 joint |
| --- | --- |
| 0 | `J00_HIP_PITCH_L` |
| 1 | `J01_HIP_ROLL_L` |
| 2 | `J02_HIP_YAW_L` |
| 3 | `J03_KNEE_PITCH_L` |
| 4 | `J04_ANKLE_PITCH_L` |
| 5 | `J05_ANKLE_ROLL_L` |
| 6 | `J06_HIP_PITCH_R` |
| 7 | `J07_HIP_ROLL_R` |
| 8 | `J08_HIP_YAW_R` |
| 9 | `J09_KNEE_PITCH_R` |
| 10 | `J10_ANKLE_PITCH_R` |
| 11 | `J11_ANKLE_ROLL_R` |
| 12 | `J12_TORSO_YAW` |
| 13 | `J13_SHOULDER_PITCH_L` |
| 14 | `J14_SHOULDER_ROLL_L` |
| 15 | `J15_SHOULDER_YAW_L` |
| 16 | `J16_ELBOW_PITCH_L` |
| 17 | `J17_ELBOW_YAW_L` |
| 18 | `J20_SHOULDER_PITCH_R` |
| 19 | `J21_SHOULDER_ROLL_R` |
| 20 | `J22_SHOULDER_YAW_R` |
| 21 | `J23_ELBOW_PITCH_R` |
| 22 | `J24_ELBOW_YAW_R` |
| 23 | `J27_HEAD_PITCH` |
| 24 | `J28_HEAD_YAW` |

Isaac articulation 的内部顺序是：

```text
J00, J06, J12, J01, J07, J13, J20, J27, J02, J08, J14, J21, J28,
J03, J09, J15, J22, J04, J10, J16, J23, J05, J11, J17, J24
```

它不能直接作为 policy/reference 的列顺序。`t800_csv_to_npz.py` 可以按内部索引写仿真状态，但保存 NPZ 时必须显式按 `T800_POLICY_JOINT_NAMES` 重新索引。

SDK 侧 0-17 与训练命名相同；训练侧右臂 `J20-J24` 对应 SDK `J18-J22`，训练侧头部 `J27-J28` 对应 SDK `J23-J24`。导出器必须使用名字映射，不能依赖数字后缀或 articulation 下标。

SDK 的 `rl_dance_example_runner` 会按 YAML 中的 `joint_names[i]` 查找真实 deploy index，因此 action 第 `i` 维仍与 NPZ/训练的第 `i` 维保持同一语义。`scripts/t800_validate_sdk_joint_contract.py` 用于同时检查 canonical NPZ 和 SDK YAML，替换 policy/trajectory 后应先运行该检查。

## 7. MotionDecode-G1 到 T800 的关节映射

这部分来自 `gmr/scripts/motiondecode_g1_csv_to_t800.py` 的 `T800_JOINT_MAP`。

| T800 joint | MotionDecode/G1 source column | sign | 说明 |
| --- | --- | --- | --- |
| `J00_HIP_PITCH_L` | `dof_left_hip_pitch_joint(rad)` | 1 | 同名语义映射 |
| `J01_HIP_ROLL_L` | `dof_left_hip_roll_joint(rad)` | 1 | 同名语义映射 |
| `J02_HIP_YAW_L` | `dof_left_hip_yaw_joint(rad)` | 1 | 同名语义映射 |
| `J03_KNEE_PITCH_L` | `dof_left_knee_joint(rad)` | 1 | G1 knee 到 T800 knee pitch |
| `J04_ANKLE_PITCH_L` | `dof_left_ankle_pitch_joint(rad)` | 1 | 同名语义映射 |
| `J05_ANKLE_ROLL_L` | `dof_left_ankle_roll_joint(rad)` | 1 | 同名语义映射 |
| `J06_HIP_PITCH_R` | `dof_right_hip_pitch_joint(rad)` | 1 | 同名语义映射 |
| `J07_HIP_ROLL_R` | `dof_right_hip_roll_joint(rad)` | 1 | 同名语义映射 |
| `J08_HIP_YAW_R` | `dof_right_hip_yaw_joint(rad)` | 1 | 同名语义映射 |
| `J09_KNEE_PITCH_R` | `dof_right_knee_joint(rad)` | 1 | G1 knee 到 T800 knee pitch |
| `J10_ANKLE_PITCH_R` | `dof_right_ankle_pitch_joint(rad)` | 1 | 同名语义映射 |
| `J11_ANKLE_ROLL_R` | `dof_right_ankle_roll_joint(rad)` | 1 | 同名语义映射 |
| `J12_TORSO_YAW` | `dof_waist_yaw_joint(rad)` | 1 | 只保留腰 yaw |
| `J13_SHOULDER_PITCH_L` | `dof_left_shoulder_pitch_joint(rad)` | 1 | 同名语义映射 |
| `J14_SHOULDER_ROLL_L` | `dof_left_shoulder_roll_joint(rad)` | 1 | 同名语义映射 |
| `J15_SHOULDER_YAW_L` | `dof_left_shoulder_yaw_joint(rad)` | 1 | 同名语义映射 |
| `J16_ELBOW_PITCH_L` | `dof_left_elbow_joint(rad)` | -1 | 肘 pitch 方向相反，需翻符号 |
| `J17_ELBOW_YAW_L` | `dof_left_wrist_yaw_joint(rad)` | 1 | 用 G1 wrist yaw 近似 T800 elbow yaw |
| `J20_SHOULDER_PITCH_R` | `dof_right_shoulder_pitch_joint(rad)` | 1 | 同名语义映射 |
| `J21_SHOULDER_ROLL_R` | `dof_right_shoulder_roll_joint(rad)` | 1 | 同名语义映射 |
| `J22_SHOULDER_YAW_R` | `dof_right_shoulder_yaw_joint(rad)` | 1 | 同名语义映射 |
| `J23_ELBOW_PITCH_R` | `dof_right_elbow_joint(rad)` | -1 | 肘 pitch 方向相反，需翻符号 |
| `J24_ELBOW_YAW_R` | `dof_right_wrist_yaw_joint(rad)` | 1 | 用 G1 wrist yaw 近似 T800 elbow yaw |
| `J27_HEAD_PITCH` | none | 0 | 源数据无对应列，置 0 |
| `J28_HEAD_YAW` | none | 0 | 源数据无对应列，置 0 |

## 8. tracking NPZ 格式

`whole_body_tracking/scripts/t800_csv_to_npz.py` 在 IsaacLab 中 replay GMR/T800 motion，并导出：

- `fps`
- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`
- `joint_names`
- `joint_order_version`
- `body_names`

其中 root state 来自输入 `root_pos/root_rot`，joint state 来自 25 DoF `dof_pos`。脚本只 render/replay，不执行动力学 step；它调用 `sim.render()` 而不是 `sim.step()`，目的是把参考运动转换成训练命令需要的完整 body state。

加载时 `MotionLoader` 按 `joint_names` 重排到 `t800_policy_v1`。没有元数据的旧 T800 NPZ 只能按 `T800_LEGACY_ISAAC_STORAGE_JOINT_NAMES` 迁移，不能默认认为它已经是 policy 顺序。迁移与验证脚本分别为：

- `scripts/t800_migrate_tracking_npz_joint_order.py`
- `scripts/t800_validate_joint_contract.py`

本次排查确认，旧联合策略不稳定的主要原因之一就是旧 NPZ 的 `joint_pos/joint_vel` 使用 articulation 拓扑顺序，而 action、SDK 和导出元数据按语义顺序解释，造成腿、双臂和头部列错位。

## 9. 恢复动作链拼接

脚本：`whole_body_tracking/scripts/t800_build_recovery_chain_candidates.py`

当前通过审计的恢复动作：
`recovery_supine_male2_reverse_fall_to_ready`

拼接逻辑：

1. 读取 GMR pkl：`root_pos`, `root_rot`, `dof_pos`, `fps`。
2. 对倒地片段执行 reverse，使其从地面姿态回到蹲姿。
3. 读取 `transition_male2_crouch_to_ready_stageii.pkl` 作为蹲姿到站立 ready 的桥接。
4. 用第一段末帧和第二段 anchor 帧的 yaw 差计算 `delta_q`。
5. 将第二段 root trajectory 旋转并平移到第一段末帧。
6. 可在 transition 前若干帧里搜索最接近的 splice anchor，评分为 `dof_rms + 2.0 * z_error`。
7. 中间用 dof/root_pos/root_rot 线性或 overlap blend 平滑连接。
8. 开头和结尾分别插入 hold，便于训练时看到明确初始和完成状态。

这一步是为了补足“完整跌倒恢复”，属于运动片段后处理，不属于 GMR IK 本体。

## 10. 已审计通过并进入训练的入围赛动作

| motion | 类别 | 对应要求 | 时长 |
| --- | --- | --- | --- |
| `kick_push_left_g17_stageii` | official mimic | 正蹬/前蹬候选 | 4.55 s |
| `kick_reverse_spin_cresent_right_g20_stageii` | official mimic | 回旋踢候选 | 2.85 s |
| `punch_cross_left_e3_stageii` | official mimic | 直拳候选 | 3.27 s |
| `punch_hook_left_e5_stageii` | official mimic | 摆拳/勾拳候选 | 2.73 s |
| `punch_jab_left_e1_stageii` | official mimic | 直拳变体 | 3.33 s |
| `recovery_supine_male2_reverse_fall_to_ready` | recovery | 躺姿恢复到站立 ready | 9.27 s |

对应 manifest：
`whole_body_tracking/configs/t800_qualifier_approved_20260902.json`

## 11. 可视化审计规则

参考视频应优先使用 T800 visual MJCF/mesh 渲染，而不是只看 collision geom：

- MuJoCo/GMR visual renderer：`gmr/scripts/render_motiondecode_t800_videos.py`
- Sonic/MuJoCo mode switch renderer：`GR00T-WholeBodyControl/gear_sonic/scripts/render_t800_mode_switch.py`

如果只看到碰撞体，通常是 renderer 使用了 collision XML 或没有加载 `t800_visual.xml`/mesh。若 mesh 方向错，应先检查 MJCF mesh frame 和 link frame，而不是直接调 policy。

## 12. 当前流程的限制

- GMR 输出是运动学可达参考，不保证机器人在动力学仿真中稳定。
- 稳定性由 IsaacLab/BeyondMimic policy 训练、reward 和 PD/actuator 参数决定。
- MotionDecode-G1 直接映射会丢失 G1 上 T800 没有的 DoF，尤其是 waist roll/pitch 和 wrist pitch/roll。
- 腿陷地问题通常来自 root-z 后处理或渲染模型不一致，需要用同一个 T800 FK/collision/visual 模型闭环检查。
- 训练前必须先渲染参考动作做人工审计；不正确的参考动作会直接污染 mimic policy。
- 参考视频只验证运动学外观，不等于动力学 policy 已稳定。单动作必须通过完整时长的物理 rollout 验收后才能进入联合训练。
- 跌倒起身的初始状态分布和五个站立动作不同，当前保留独立 policy；五个站立动作达标后再训练 `stand -> action -> stand` 联合切换策略。

## 13. 复现实用命令

官方 mimic 候选重定向和渲染：

```bash
cd /data2/yangky/test/whole_body_tracking
PYTHON_BIN=/data2/yangky/miniconda3/envs/gmr/bin/python \
  scripts/t800_retarget_official_mimic_candidates.sh
```

恢复动作链生成：

```bash
cd /data2/yangky/test/whole_body_tracking
/data2/yangky/miniconda3/envs/gmr/bin/python \
  scripts/t800_build_recovery_chain_candidates.py \
  --only recovery_supine_male2_reverse_fall_to_ready \
  --transition-search-frames 30 \
  --blend-mode overlap
```

GMR/T800 motion 转 tracking NPZ：

```bash
cd /data2/yangky/test/whole_body_tracking
/data2/yangky/miniconda3/envs/env_isaaclab/bin/python \
  scripts/batch_prepare_t800_motions.py \
  --manifest configs/t800_qualifier_approved_20260902.json \
  --force \
  --headless
```

验证 canonical 关节契约：

```bash
cd /data2/yangky/test/whole_body_tracking
/data2/yangky/miniconda3/envs/gmr/bin/python \
  scripts/t800_validate_joint_contract.py \
  artifacts/t800_approved_qualifier_20260902/tracking_npz_canonical_v1/*.npz
```

验证 canonical NPZ 与 EngineAI SDK 配置的逐位置语义：

```bash
cd /data2/yangky/test/whole_body_tracking
/data2/yangky/miniconda3/envs/env_isaaclab/bin/python \
  scripts/t800_validate_sdk_joint_contract.py \
  --npz artifacts/t800_approved_qualifier_20260902/tracking_npz_canonical_v1/punch_cross_left_e3_stageii_tracking.npz \
  --sdk-config ../engineai_robotics_native_sdk/assets/config/t800/rl_qualifier_approved_20260902/qualifier_straight_punch.yaml
```

在 sys01 后台启动“单动作验收后再联合”的训练队列：

```bash
cd /mnt/data/yangky/test/whole_body_tracking
GPU=1 NUM_ENVS=512 EVAL_ENVS=64 EVAL_EPISODES=5 MIN_SUCCESS_RATE=0.95 \
  scripts/start_t800_qualifier_staged_train_tmux.sh
```

该队列会分别训练五个站立动作和独立 recovery policy。每轮先检查 TensorBoard 收敛指标，再执行默认 320 次物理 rollout；成功率达到 95% 才生成 H264 视频并允许该动作进入后续联合切换训练。

联合站立动作和独立 recovery 都通过后，后台 watcher 会调用 `scripts/t800_stage_engineai_sdk_assets.py`，生成：

- `engineai_robotics_native_sdk/assets/config/t800/rl_qualifier_canonical_v1_20260902`
- `engineai_robotics_native_sdk/assets/config/t800/mode_qualifier_canonical_v1.yaml`
- `engineai_robotics_native_sdk/assets/config/t800/task_motion/qualifier_canonical_v1.yaml`

只有目录中存在 `READY.json` 才允许启动 canonical SDK 场景：

```bash
cd /mnt/data/yangky/test/engineai_robotics_native_sdk
./scripts/start_t800_qualifier_canonical_mujoco_tmux.sh
```

五个站立动作共用通过验收的 joint policy，但分别加载自己的 `stand -> action -> stand` trajectory；躺姿起身加载独立 recovery policy。SDK 仍通过原有按键/键盘映射切换 runner 的 `param_tag`。

同步审计通过资产、训练 ckpt、policy、视频到 HF staging：

```bash
cd /data2/yangky/test
PUSH=0 scripts/sync_t800_qualifier_assets_to_hf.sh
```
