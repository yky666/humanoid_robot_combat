# T800 Canonical V1 Results

Snapshot date: 2026-09-05 (Asia/Shanghai)

## Accepted Policies

| Key | Backend | Result | Policy location |
| --- | --- | ---: | --- |
| `hook_punch` | IsaacLab/RSL-RL | 320/320 | `policies/hook_punch/` |
| `front_kick` | IsaacLab/RSL-RL | 320/320 | `policies/front_kick/` |
| `straight_punch` | IsaacLab/RSL-RL | 320/320 | `policies/straight_punch/` |
| `jab_left` | IsaacLab/RSL-RL | 316/320 | `policies/jab_left/` |
| `recovery_supine` | EngineAI Native SDK MNN | 320/320 | `recovery_official/` |

The authoritative acceptance metadata is under `qualification/best/`. Absolute
paths in those original JSON reports record their sys01 provenance; repository
copies of the referenced files are organized here under relative paths.

## Failed/Incomplete Work

`spinning_kick_r9_training.json` reports a 95.13% timeout rate and 4.58%
end-effector termination rate. The required limits are at least 95% and at most
3%, respectively, so the motion failed before the 320-rollout stage.

The five-standing-motion joint policy was not trained because spinning kick did
not pass. No joint checkpoint in this archive should be inferred or fabricated.

## Contents

- `approved_reference/`: approved GMR PKL/NPZ, canonical tracking NPZ, manifests,
  and reference H.264 videos.
- `checkpoints/final_rounds/`: one terminal checkpoint per training report,
  including failed rounds for diagnosis and continuation.
- `policies/`: accepted checkpoints, ONNX exports, training parameters, and the
  WBT diff captured by each accepted run.
- `qualification/`: training-tail checks, formal rollout reports, review frames,
  and accepted `best` manifests.
- `media/reference_reviews/`: all retained WBT reference-review videos.
- `media/policy_renders/`: every MP4 retained below the T800 RSL-RL log tree.
- `media/artifact_results/`: every MP4 retained below the WBT artifact tree.
- `archives/`: complete compressed logs and auxiliary experiment trees.
- `recovery_official/`: official MNN policy, trajectory, and config used for the
  accepted supine-to-stance recovery.

Run `git lfs pull` before attempting playback or policy loading.
