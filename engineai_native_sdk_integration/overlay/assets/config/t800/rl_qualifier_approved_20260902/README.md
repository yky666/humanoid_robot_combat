# T800 Qualifier Approved EngineAI SDK Staging

This directory stages manually audited T800 URKL qualifier trajectories for the EngineAI SDK `rl_dance_example_runner`.

Use `assets/config/t800/mode_qualifier_approved.yaml` as the candidate mode map and `assets/config/t800/task_motion/qualifier_approved.yaml` as the candidate task state machine.

The current `rl_dance_example_runner` now supports either MNN or ONNXRuntime inference. The staged qualifier YAML files point to `rl_qualifier_approved_20260902/policies/t800_qualifier_joint_policy.onnx`, copied from the accepted IsaacLab joint policy export.

The trajectory columns follow the same 25-DoF semantic order as the IsaacLab/GMR policy. The joint names in these YAML files follow the EngineAI SDK T800 MuJoCo XML naming, where the right arm is `J18..J22` and the head is `J23/J24`.
