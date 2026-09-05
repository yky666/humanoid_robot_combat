# T800 qualifier real-robot deployment

This directory packages the accepted six-motion joint policy and 50 Hz reference trajectories for the independent
T800 deployment dated 2026-09-04. The runtime consumes an actor-only MNN model with contract
`obs[1,140] -> actions[1,25]`; the original training-export ONNX is retained only for provenance.

Real-robot guards are configured per motion: action clipping, joint-limit margin, initial-pose rejection and a
startup blend that holds reference frame zero until interpolation completes. Standing motions are reachable only
from `pd_stand`; `qualifier_recovery_supine` is reachable only from `passive`. See
`task_motion/qualifier_robot.yaml` for the restricted state graph.

Do not start this controller until the official controller has been stopped deliberately and the robot is secured
on its stand/harness, the area is clear, the battery is adequate, and an operator is holding the emergency stop.
