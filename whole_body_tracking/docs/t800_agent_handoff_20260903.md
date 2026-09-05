# T800 qualifier project handoff (2026-09-03 16:03 CST)

## Goal and decisions

- Retarget audited martial-art motions to EngineAI T800, train stable IsaacLab policies, render all accepted results, and deploy keyboard-selectable policies in the EngineAI native SDK MuJoCo simulator.
- EngineAI SDK is the primary deployment path. Sonic/GR00T is a secondary experiment only; its current T800 mode-switch probe is unstable and is not accepted evidence.
- Train five standing actions independently first. Train supine recovery as an independent policy. Only after all five standing actions pass should a stand/action/stand joint switching policy be trained.
- A policy is accepted only after a 320-rollout physical evaluation (64 environments x 5 batches) reaches at least 95% success. TensorBoard tail metrics alone are not acceptance.

## Hosts and roots

- Local root: `/data2/yangky/test`
- Remote training host: `sys01@100.74.87.113`
- Remote root: `/mnt/data/yangky/test`
- Local WBT: `/data2/yangky/test/whole_body_tracking`
- Remote WBT: `/mnt/data/yangky/test/whole_body_tracking`
- IsaacLab Python on sys01: `/home/sys01/miniconda3/envs/env_isaaclab/bin/python`
- Training uses physical GPU1 via `CUDA_VISIBLE_DEVICES=1`; commands then use logical `cuda:0`.
- Do not include SSH/Hugging Face credentials in commits or documentation. Credentials already exist in the environment.

## Approved references

Manifest: `whole_body_tracking/configs/t800_qualifier_canonical_v1_20260902.json`

All canonical tracking files are under:

`whole_body_tracking/artifacts/t800_approved_qualifier_20260902/tracking_npz_canonical_v1/`

Audited motions:

| Policy label | Reference |
|---|---|
| `front_kick` | `kick_push_left_g17_stageii_tracking.npz` |
| `spinning_kick` | `kick_reverse_spin_cresent_right_g20_stageii_tracking.npz` |
| `straight_punch` | `punch_cross_left_e3_stageii_tracking.npz` |
| `hook_punch` | `punch_hook_left_e5_stageii_tracking.npz` |
| `jab_left` | `punch_jab_left_e1_stageii_tracking.npz` |
| `recovery_supine` | `recovery_supine_male2_reverse_fall_to_ready_tracking.npz` |

- Canonical reference videos: `whole_body_tracking/review_videos/t800_reference_canonical_v1_20260902/`
- Older videos under `review_videos/t800_policy` and old Sonic probe videos are diagnostics only and must not be treated as accepted policy output.

## Joint contract

- Canonical order version: `t800_policy_v1`, 25 joints.
- Source of truth: `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/t800_joint_order.py`
- Retargeting explanation: `whole_body_tracking/docs/t800_retargeting_logic_20260902.md`
- Validators:
  - `whole_body_tracking/scripts/t800_validate_joint_contract.py`
  - `whole_body_tracking/scripts/t800_validate_sdk_joint_contract.py`
- All six approved NPZ files pass the canonical joint-order validator and raw first-frame comparison with max error 0.

## Current live training

- Remote tmux: `t800_qualifier_staged_train`
- Remote queue log:
  `/mnt/data/yangky/test/whole_body_tracking/artifacts/t800_qualifier_training_canonical_v1_20260902/logs/queue_gatefix_20260903_1316.log`
- State root:
  `/mnt/data/yangky/test/whole_body_tracking/artifacts/t800_qualifier_training_canonical_v1_20260902/`
- Snapshot at 2026-09-03 16:03 CST:
  - Active motion: `hook_punch`, round `r5`
  - Iteration: `21682/23799`
  - Latest checkpoint: `.../2026-09-03_15-18-50_canonical_v1_hook_punch_r5/model_21600.pt`
  - GPU1: about 13.7 GiB used, 10.4 GiB free, 58% utilization
- `r5` resumed from `r4/model_18799.pt`.
- `r4` training-tail result failed only the end-effector termination gate:
  - timeout rate `0.95274` (pass)
  - mean episode length `487.01` (pass)
  - anchor-position termination `0.00366` (pass)
  - end-effector termination `0.04360` (fail; maximum `0.03`)
  - joint-position error `0.78735` (pass)
- The old `hook_punch r2/model_9998.pt` was re-evaluated correctly: `0/320`, success rate `0.0`; it is rejected.
- `best/` is currently empty. No action has passed the formal physical gate yet.

Training order in the active queue:

1. `hook_punch` (currently active)
2. `front_kick`
3. `spinning_kick`
4. `straight_punch`
5. `jab_left`
6. independent `recovery_supine`
7. joint stand/action/stand policy, only if all five standing actions passed

Configuration: 512 training environments, 5000 iterations per round, up to 6 additional standing-action rounds, 64 x 5 evaluation trials, 95% minimum success.

Useful checks on sys01:

```bash
tmux attach -t t800_qualifier_staged_train
tail -f /mnt/data/yangky/test/whole_body_tracking/artifacts/t800_qualifier_training_canonical_v1_20260902/logs/queue_gatefix_20260903_1316.log
nvidia-smi -i 1
```

## Fixed blockers

1. Canonical NPZ, IsaacLab, and EngineAI SDK joint orders were unified with explicit named mappings.
2. `evaluate_t800_policy.py` no longer advances IsaacLab under `torch.inference_mode()`, which previously caused an in-place inference-tensor crash on the second reset.
3. The Hydra decorator discarded the evaluator return value, and Isaac Sim shutdown forced exit code 0. The evaluator now stores the gate result explicitly and uses a flushed `os._exit(0|2)` so shell automation receives the real status. Both pass and fail exit paths were tested.
4. The scene uses a local generated flat mesh and local preview material. This removed the remote Omniverse/S3 `default_environment.usd` dependency that blocked spinning-kick and straight-punch startup.
5. The staged queue resumes the latest existing checkpoint, supports an explicit standing-motion order, and renders/copies `best` only after physical evaluation passes.
6. HF synchronization now publishes only `best/*.json` whose physical report has `status: passed`.
7. The HF watcher now honors an externally supplied queue log path.

Modified files relevant to these fixes:

- `whole_body_tracking/scripts/rsl_rl/evaluate_t800_policy.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `whole_body_tracking/scripts/t800_qualifier_staged_train_queue.sh`
- `scripts/sync_t800_canonical_assets_to_hf.sh`
- `scripts/watch_t800_canonical_train_and_hf_upload.sh`

## Hugging Face state

- Dataset: `yky666/engineai_urkl_sysu_frontier_fighter`
- Canonical subtree: `urkl_qualifier_canonical_v1_20260902/`
- Canonical NPZ files, reference videos, manifests, retargeting docs, validators, and reports have been uploaded.
- A failed hook-punch r2 checkpoint was briefly misclassified because of the evaluator exit-code bug. Its checkpoint, ONNX, params, videos, and best marker were removed again in commit `647206a75c8727cf0584f6867da98599f9e225fc`.
- A recoverable local copy is quarantined at:
  `/data2/yangky/test/hf_upload_quarantine/20260903_failed_hook_punch_r2_policy/`
- Local HF watcher tmux: `t800_canonical_hf_watch`
- Watcher log: `/data2/yangky/test/logs/t800_canonical_gatefix_hf_watch.log`
- On queue completion, the watcher stages SDK assets only if `best/joint_standing5.json` exists, then performs the final HF API upload.

## EngineAI SDK and Sonic status

- Existing remote EngineAI SDK tmux: `t800_qualifier_mujoco`.
- Existing qualifier MuJoCo switching is not accepted yet because it used unverified/unstable policies; falls in that simulator are expected until accepted checkpoints are staged.
- Staging script: `whole_body_tracking/scripts/t800_stage_engineai_sdk_assets.py`.
- SDK policy assets must not be activated until the corresponding physical evaluation report passes. Joint-policy staging is gated on `best/joint_standing5.json`.
- Final EngineAI work still needs keyboard locomotion plus explicit transitions: stable stand -> selected action -> stable stand, with recovery kept separate.
- Sonic/GR00T T800 adaptation has only a probe/demo status; no verified T800 planner encoder/decoder has passed stability validation. Keep it secondary to the SDK route.

## TODO in priority order

1. Keep `t800_qualifier_staged_train` alive; monitor r5 completion and its training-tail report.
2. If the training tail passes, confirm the evaluator runs all 320 trials and requires at least 304 successes. Do not accept a video or TensorBoard metric alone.
3. Continue hook-punch rounds from the newest checkpoint until the physical gate passes or all configured rounds are exhausted. Diagnose end-effector terminations if r5 remains above 3%.
4. Let the queue re-evaluate the previous front-kick checkpoint with the repaired evaluator, then continue training as needed.
5. Train/accept spinning kick, straight punch, and jab independently. The local terrain fix should allow spinning/straight startup now.
6. Train and evaluate supine recovery as an independent policy; render only after it passes.
7. Build and train stand/action/stand references and the five-motion joint policy only after all five independent standing policies pass.
8. Render H.264 MuJoCo/IsaacLab videos for every accepted policy and the final mode-switch sequence; manually audit balance, foot contact, action identity, and return-to-stand.
9. Stage accepted assets into EngineAI SDK, validate the 25-joint contract again, and test keyboard WASD locomotion plus action keys in MuJoCo.
10. Verify the final HF tree contains accepted checkpoints/ONNX/params/videos/reports and contains no rejected policy under `training/best` or `training/policies`.

## Guardrails for the next agent

- Do not call old TensorBoard-tail reports "passed policies". `best/*.json` plus a 95% physical report is the contract.
- Do not train from unaudited motions or old policy videos.
- Do not change the canonical joint order implicitly; use named permutations and run both validators.
- Do not restart from scratch while a newer checkpoint exists. Resume the newest valid checkpoint.
- Do not stop unrelated GPU0 jobs. This project is assigned to physical GPU1.
- Do not upload secrets. Use existing local credentials and HF API upload; previous git/LFS push attempts could hang.
