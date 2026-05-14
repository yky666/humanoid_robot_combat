# humanoid_robot_combat

This repository bundles the two local projects used in the current T800 humanoid combat-motion pipeline:

- `whole_body_tracking`
  Isaac Lab training, motion conversion, smoke testing, and policy playback.
- `GMR`
  General Motion Retargeting scripts and robot retarget configs used to convert source motions into T800-compatible inputs.

## Layout

```text
humanoid_robot_combat/
  README.md
  whole_body_tracking/
  GMR/
```

## What Is Intentionally Excluded

The local working trees contain large generated artifacts that are not copied into this repository snapshot:

- training logs
- wandb runs
- motion artifacts and cached outputs
- large checkpoints and media
- temporary caches

Those outputs remain in the original local paths under `/data2/yangky/test/...`.

## Current Focus

The current engineering focus is T800 combat motion conversion and training, including:

- BVH to T800 retargeting for local motions such as `540huixuantitui_001.bvh` and `zhiquan_quanji_001.bvh`
- legacy local `.npy` combat motions such as `riot_combo.npy`
- smoke validation and formal PPO training for motions that remain stable
