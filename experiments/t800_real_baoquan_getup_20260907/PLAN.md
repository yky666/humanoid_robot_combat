# T800 Real Baoquan Get-Up Target Plan

Date: 2026-09-07

Objective: ingest real T800 joint logs, verify the measured boxing-guard pose in MuJoCo, then use the approved terminal pose for prone/supine direct-RL get-up training.

## Current Inputs

- Source archive: `/mnt/data/yangky/test/datasets/urkl_locomotion_260901/motion_logs.zip`
- Primary guard transition: `logs/pdstand2baoquan.csv`
- Auxiliary transitions:
  - `logs/baoquan2pdstand.csv`
  - `logs/prone2pdstand.csv`
  - `logs/supine2pdstand.csv`

## Method

1. Parse SDK-order joint logs and map them to canonical T800 policy/MuJoCo joint order.
2. Extract a robust measured boxing-ready target from the stable tail of `pdstand2baoquan.csv`.
3. Render `pdstand2baoquan` in MuJoCo with a fixed upright root for visual inspection.
4. After visual approval, replace the direct get-up placeholder target with the measured target.
5. Launch short prone/supine simulation smoke runs before any longer training.
6. Only stage/export/deploy policies that pass rollout inspection and ONNX/MNN contract checks.

## Notes

These logs contain joint position, velocity, and torque only. They do not contain global root pose or IMU orientation, so the MuJoCo replay is a kinematic joint-pose inspection, not a physics validation of the recovery trajectory.
