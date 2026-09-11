# T800 训练与部署备忘（2026-09-11）

本文整理当前步态训练、过往动作实验、以及真机/SDK 部署经验。
正式操作流程仍以 [Real-Robot Deployment](REAL_ROBOT_DEPLOYMENT.md) 为准。

竞赛包装约束见
[Competition Compliance Audit](T800_COMPETITION_COMPLIANCE_AUDIT_20260909.md)：
**最终上机控制器不能是官方 locomotion / official mimic**。
训练阶段可以用官方走步网当 teacher prior，但导出给比赛的必须是自己的 student policy。

---

## 1. 当前进度（FG72 抱拳 + 步态）

截至 2026-09-11 18:00 CST，sys01 上仍在双卡训练：

| 项 | 值 |
| --- | --- |
| Run | `t800_fg72_prior_bump_1310` |
| Checkpoint | `model_15400.pt`（约 iter 15409） |
| 任务 | `Tracking-Bump-T800-Fixed-Guard-72-v0` |
| 阶段 | 6 `speed_bump`，指令模式 `back_yaw`，command_scale 0.55 |
| Reward / ep_len | ~124 / ~990 |
| `error_vel_xy` / `error_vel_yaw` | ~0.26 / ~0.34（目标满杆 ≤0.15 / ≤0.18） |
| 存活 | `timeout≈0.98`，`bad_ori≈0.02` |
| Teacher mix | ~0.66（0.88 → 0.30 退火中） |
| 根高度 | 约 0.91–0.98 m（目标 1.00 m，仍略低） |
| TensorBoard | **http://100.74.87.113:6007**（`t800_fixed_guard_velocity_72d`） |

6006 是旧的起身训练（`t800_flat`），不要当成当前步态。

Isaac play 视频目录（本机曾拷贝）：`t800_videos/bump_*.mp4`。
最新权重比 16:30 的 `model_13300` 又训了约 2000 iter，视频会落后于当前 ckpt。

---

## 2. 现在在训什么

一条 **22-DOF** 策略：12 腿 + 10 臂，不含躯干/头。

- 上半身锁抽帧抱拳（`measured_boxing_ready` / `T800_BAOQUAN_POLICY_POSE`）
- 下半身用虚拟摇杆走：当前操作约定是 **后退 + 转向**，侧移容易剪刀步
- 观测对齐官方 SDK 走步合同：`72 × 15` 历史 + 3 维命令尾 = **1083**
- 动作：`q_des = q_pd + action * action_scale`，scale 与 `T800_ACTION_SCALE` 一致

Gym：

| ID | 用途 |
| --- | --- |
| `Tracking-Flat-T800-Fixed-Guard-72-v0` | 平地 72-D |
| `Tracking-Rough-T800-Fixed-Guard-72-v0` | ≤10 cm 粗糙 + 小盒子 |
| `Tracking-Bump-T800-Fixed-Guard-72-v0` | **当前**：平地 / 梯形减速带 / 轻粗糙 |
| `*-Play-v0` | 回放；用 `FG_PLAY_VX/VY/WZ` 模拟摇杆 |

减速带截面（越障主目标）：底边 350 mm、顶边 100 mm、高 70 mm。
地形混合约 30% 平地 / 50% 减速带 / 20% 轻粗糙，出生点留平坦垫。

关键代码：

- `whole_body_tracking/source/.../tasks/fixed_guard/`
- `utils/official_walk_prior.py`
- `utils/sdk_observation.py`
- `scripts/pipeline/fg72_track_watchdog.py`

---

## 3. 有效方法

1. **官方走步当腿部 action prior，不要把 Isaac obs 直接喂进 MNN。**
   官方网的零点、速度 0.05 缩放、抱拳臂都和 Isaac 学生策略不同。
   Wrapper 从仿真重建官方 72-D 帧：腿相对官方 default，臂在 teacher 观测里伪装成官方默认臂。
2. **腿上叠加轻蹲，teacher 观测再“还原”成官方站姿。**
   PD 零点髋/膝/踝约 −0.20 / 0.34 / −0.14，出生高度 1.02 m，高度奖励目标 1.00 m。
   `delta_leg = 0`，teacher residual 直接加在蹲姿 PD 上，重心比官方直立走更低。
3. **臂动作强制 0，保持抱拳。** `FG72_WALK_PRIOR_ARM_KEEP=1`。
4. **侧移先关掉。** 横向最容易交叉穿模；`command_mode=back_yaw`：
   `vx ∈ [-0.70s, 0.12s]`，`vy≈0`，`yaw ∈ ±0.85s`。
5. **脚距用机体系，不要用世界系 Y。** 转向后世界 Y 没有意义。
   `feet_y_distance` 惩罚步宽偏离、并脚、左右脚同侧（剪刀）。
6. **课程状态放 JSON，DDP 两卡读同一份。** `FG72_STATE_JSON`。
   存活门限 ≠ 会走：`ep_len=1000` 只说明没倒，`error_vel_xy` 才是跟踪。
7. **双卡 DDP，每卡 1024、合计 2048 env。** 回放可占 GPU1 剩余显存（训练大约 5–6 GB/卡）。
8. **回放不要开 `FG_PLAY_WIDE=1`。** 世界相机在地形块上经常拍空。用默认 follow、1 个 env。

环境变量（训练）：

```bash
export FG72_WALK_PRIOR=1
export FG72_WALK_PRIOR_MIX=0.88          # 当前 run 已退火到 ~0.66
export FG72_WALK_PRIOR_MIX_END=0.30
export FG72_WALK_PRIOR_ANNEAL_STEPS=200000
export FG72_WALK_PRIOR_ARM_KEEP=1.0
export FG72_WALK_PRIOR_IMIT=0.08
export FG72_CMD_MODE=back_yaw
export FG72_STATE_JSON=.../pipeline/state.json
```

Teacher MNN（只作训练 prior，不是比赛控制器）：

`engineai_robotics_native_sdk/assets/config/t800/rl_walking_example/policies/251111_180036_saw_50k.mnn`

---

## 4. 踩过的坑（不要重复）

| 做法 | 结果 |
| --- | --- |
| 从抱拳蹲纯 RL 学走 | 原地抱拳局部最优，`err_xy≈0.5`，看起来“活着”但跟不住杆 |
| Isaac 72-D 观测直接进官方 MNN | 零点/速度尺度/抱拳臂全 OOD |
| mix=1 + 抱拳蹲出生 | Teacher 立即失败 |
| 用世界系脚间距 | 一 yaw 就奖错方向 |
| 大力侧移 | 剪刀步、脚交叉穿模 |
| 存活课程当走路课程 | stage 5 过了仍不会走 |
| 抱拳腿姿态奖励 vs 官方步态 | 两项对打，又缩回原地蹲 |
| 把 12-DOF `baoquan_locomotion` 或旧 get-up ckpt 续进 FG72 | 合同不同，不要续 |
| TensorBoard 6006 | 起身，不是 FG72 |
| 仰卧桥 / collapsed 12-DOF | 已废弃 |

不要续训：旧站立 `model_900`、12-DOF `model_700`、get-up/bridge 权重。

---

## 5. 课程阶段

Watchdog JSON 的 `stage`：

| stage | 名字 | 内容 |
| ---: | --- | --- |
| 0 | standing_balance | 命令全 0，先站住 |
| 1 | forward_warmup | 小前进 |
| 2 | backward_expansion | 加入后退 |
| 3 | lateral_expansion | 侧移（当前 back_yaw 模式下后期不再用） |
| 4 | yaw_mixed | 转向 |
| 5 | light_terrain | ≤10 cm 粗糙 + 小盒子 |
| 6 | speed_bump | **当前** 梯形减速带 |

跟踪门限（`command_scale`，与 stage 独立）：

0.55 → 0.70 → 0.85 → 1.00，满杆 `err_xy≤0.15` 才算跟踪过关。

---

## 6. 过往动作训练（简谱）

这些都还在仓库时间线里，但 **不是当前 FG72 步态**。

### 6.1 Tracking / BeyondMimic 风格

GMR 重定向 AMASS/BVH → T800 tracking NPZ → Isaac PPO。
资格门限历史上是 64×5=320 rollouts、成功率 ≥0.95。

| 动作 | 状态 |
| --- | --- |
| 勾拳 / 前踢 / 直拳 | 仿真门限过；若禁止官方 mimic 则竞赛不干净 |
| 左 jab | 316/320 |
| 旋转/回旋踢 | 未过门限 |
| 官方仰卧起身 MNN | 仅 debug/参考 |
| 站立-动作-站立 联合策略 | 当时未完成 |

细节：`results/t800_canonical_v1_20260902/`，以及 README 资格表。

### 6.2 真机抽帧抱拳起身（2026-09-07 后）

目标：从仰卧/俯卧爬起到 **抽帧格斗抱拳**，不是官方 `pd_stand`。

- v31 guard：800 iter 量级，护卫稳定奖励
- v32 curriculum：分阶段起身
- v33–v36：先站稳再动作；v36 从零/从 v34 分叉
- 关键教训：上半身和腿都要用抽帧抱拳零点；用官方直立腿当 get-up 零点会倒

日志在 `results/t800_real_baoquan_getup_20260907/`（体积大，默认不进 git）。
TensorBoard：6006 / `logs/rsl_rl/t800_flat`。

### 6.3 12-DOF 抱拳走 + 数据集自训（2026-09-10）

曾尝试 URKL locomotion clip + 抱拳上肢，观测合同不是 SDK 72-D。
数据集：可用移动 clip 很少，右平移几乎没有位移。
**已停止，改走 FG72 + 官方 prior。**
见 [Baoquan Locomotion Self-Training](T800_BAOQUAN_LOCOMOTION_SELF_TRAINING_20260910.md)。

---

## 7. 真机与 SDK 部署

完整 runbook：[REAL_ROBOT_DEPLOYMENT.md](REAL_ROBOT_DEPLOYMENT.md)。
录包/转 NPZ：[t800_real_motion_recording.md](../whole_body_tracking/docs/t800_real_motion_recording.md)。

现状摘要：

- 机器人 EngineAI T800，控制器 Nezha ARM64 + ROS 2 Humble
- 不覆盖厂商 `/apps/engineai_robotics`，用独立 overlay 包
- 已部署过的 tracking 合同是 **`obs[1,140] → actions[1,25]`**
- **FG72 是 `1083 → 22`，还没有导出进 SDK / 真机**
- 真机上官方 walk 仍可用虚拟摇杆在 MuJoCo/SDK 里测；那测的是官方网，不是 FG72
- 虚拟摇杆：`engineai_native_sdk_integration/overlay/tools/virtual_gamepad/send_virtual_stick.py`
- ToDesk 键盘测官方走：先 `LB+A` PD 站，再 `LB+B` walk，焦点必须在 Virtual Gamepad

上机前仍要：IMU 固件、PD 入口守卫、动作结束后肩/臂能否安全回到 `pd_stand`。
直拳硬件曾出现回收接口超守卫，动到站的过渡还没闭环。

竞赛包装：FG72 student 退火到低 mix 并自己能走之后，再 export ONNX/MNN；
不要把 `251111_180036_saw_50k.mnn` 当比赛走步控制器。

---

## 8. 常用命令

训练（sys01，`env_isaaclab`）：

```bash
cd whole_body_tracking
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/rsl_rl/train.py \
  --task Tracking-Bump-T800-Fixed-Guard-72-v0 \
  --num_envs 2048 --max_iterations 25000 \
  --headless --distributed --device cuda \
  --run_name <name> --logger tensorboard \
  --resume True --load_run <dir> --checkpoint model_XXXX.pt
```

回放（训练占用双卡时用 `CUDA_VISIBLE_DEVICES=1`）：

```bash
export FG72_WALK_PRIOR=1
export FG72_WALK_PRIOR_MIX=0.66   # 与当前退火对齐
export FG_PLAY_VX=-0.40 FG_PLAY_VY=0 FG_PLAY_WZ=0
# 不要 FG_PLAY_WIDE=1
python scripts/rsl_rl/play.py \
  --task Tracking-Bump-T800-Fixed-Guard-72-Play-v0 \
  --num_envs 1 --device cuda:0 \
  --load_run 2026-09-11_16-13-49_t800_fg72_prior_bump_1310 \
  --checkpoint model_15400.pt \
  --video --video_length 700 --headless --enable_cameras
```

Watchdog：`scripts/pipeline/fg72_track_watchdog.py`（tmux `t800_fg72_watch`）。
tmux 会话名用 `=t800_fg72` 精确匹配，否则会误匹配 `t800_fg72_tb` / `_watch`。
