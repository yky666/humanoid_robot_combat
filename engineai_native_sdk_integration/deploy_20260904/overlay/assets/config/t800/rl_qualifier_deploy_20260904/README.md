# T800 qualifier real-robot deployment

This directory packages four accepted action policies and their 50 Hz reference
trajectories for the independent T800 deployment dated 2026-09-04. Each active
motion uses its own actor-only MNN model with contract `obs[1,140] ->
actions[1,25]`; see `policies/README.md` for model provenance and hashes.

The training exports retain the robot-description identifiers with reserved
gaps (`J20` for the first right-arm joint and `J27` for the first head joint).
The Native SDK model table names the same ordered joints contiguously (`J18`
and `J23`). The YAML therefore uses Native SDK identifiers while preserving
the policy's exact semantic joint order.

Real-robot guards are configured per motion: action clipping, joint-limit margin, initial-pose rejection and a
startup blend that holds reference frame zero until interpolation completes.
Standing motions are reachable only from `pd_stand` and automatically return to
it. Spinning kick and both recovery policies are deliberately unreachable. See
`task_motion/qualifier_robot.yaml` for the restricted state graph.

The state graph starts and remains in `idle`; `passive` is a damping state, not
the default. It also includes EngineAI's official `pd_stand_x` and `pd_stand_y`
fall-recovery preparation poses. These PD states remain separate from recovery
policies because no complete qualified return-to-stand chain is integrated yet.

Do not start this controller until the official controller has been stopped deliberately and the robot is secured
on its stand/harness, the area is clear, the battery is adequate, and an operator is holding the emergency stop.
