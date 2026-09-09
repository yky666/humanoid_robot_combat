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
5. After visual approval, save the stable tail segment as a joint-log baoquan
   reference action for action-group/state-machine reuse.
6. Launch short prone/supine simulation smoke runs before any longer training.
7. Start separate direct-RL runs from prone and supine preparation poses into
   the measured boxing-ready target.
8. Only stage/export/deploy policies that pass rollout inspection and ONNX/MNN contract checks.

## Run Notes

- The approved MuJoCo render of `pdstand2baoquan` is the source for the terminal boxing-guard pose and the stable baoquan reference action.
- The 2-second tail reference is stored at `results/t800_real_baoquan_getup_20260907/baoquan_tail_reference_2s.npz`.
- Direct-RL v1 (`getup_prone`, `getup_supine`, 200 iterations) did not learn a full get-up.
- Direct-RL shaped v2 (`getup_prone_shaped`, `getup_supine_shaped`, 500 iterations) learned a repeatable sit-up / semi-seated behavior, but did not reach the final standing boxing-guard success condition.
- Current direct get-up checkpoints are research artifacts only and must not be deployed to the real T800 until a rollout gate passes.

## Next Training Route

The next experiment should add a curriculum instead of only extending v2:

1. prone/supine -> seated or kneeling intermediate posture
2. seated/kneeling -> crouched PD stand
3. crouched PD stand -> measured boxing guard

The stable real baoquan tail remains the terminal action/reference. A policy can be considered for the robot only after rendered playback looks sane and the 320-rollout gate passes with the deployment observation contract checked.

## Notes

These logs contain joint position, velocity, and torque only. They do not contain global root pose or IMU orientation, so the MuJoCo replay is a kinematic joint-pose inspection, not a physics validation of the recovery trajectory.
