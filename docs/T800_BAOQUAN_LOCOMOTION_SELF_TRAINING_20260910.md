# T800 Baoquan Locomotion Self-Training

This note tracks the competition-clean locomotion route. The final locomotion policy must not use the EngineAI official loco runner as its controller.

## Data

- Movement clips: `/mnt/data/yangky/test/datasets/urkl_locomotion_260901/tracking_npz`
- Baoquan target: `results/t800_real_baoquan_getup_20260907/baoquan_tail_reference_2s.npz`
- Audit output: `results/t800_baoquan_locomotion_20260910/dataset_audit.json`
- Clean manifest: `results/t800_baoquan_locomotion_20260910/command_manifest.json`

## Current Dataset Read

- Usable movement clips: 5
- Rejected movement clips: 3
- Right strafe is currently missing after filtering because the two `右移` clips have almost no root displacement and low-height segments.
- Forward/backward labels need robot-frame heading alignment before being used as signed velocity commands, because their world-frame displacement points in a similar direction.

## Training Direction

The target task should be a new command-conditioned T800 environment:

- observation includes proprioception, projected gravity, previous action, and command `[vx, vy, yaw_rate]`
- upper-body reward holds measured baoquan guard
- lower-body reward tracks commanded base velocity and stable foot contact
- safety rewards penalize joint limit margin, high velocity, high torque, abrupt action changes, and low base height
- movement clips are used as lower-body style/seed data only

The current official SDK walking entry remains debug-only and should not be used as final competition locomotion.

**Update 2026-09-11:** this 12-DOF / clip-seed route was superseded by FG72 + official walk prior. See [T800_TRAINING_PLAYBOOK_20260911.md](T800_TRAINING_PLAYBOOK_20260911.md).
