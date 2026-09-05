# T800 Qualification Basic Motion Audit - 2026-09-02

Source rule checked: EngineAI URKL tournament rule page. The play-in basic-score order is recovery, locomotion, then mimic. Basic mimic is limited to official straight punch, hook/swing punch, front/push kick, and roundhouse/spinning kick reference data.

## Audit Outputs

- Official mimic candidate videos: `/data2/yangky/test/whole_body_tracking/review_videos/t800_reference_audit_20260902/official_mimic_candidates/index.html`
- Full recovery-chain candidate videos: `/data2/yangky/test/whole_body_tracking/review_videos/t800_reference_audit_20260902/recovery_chains_yaw_aligned/index.html`
- Video overview sheets:
  - `/data2/yangky/test/whole_body_tracking/review_videos/t800_reference_audit_20260902/contact_sheets/official_mimic_sequence_overview.jpg`
  - `/data2/yangky/test/whole_body_tracking/review_videos/t800_reference_audit_20260902/contact_sheets/recovery_chain_yaw_aligned_sequence_overview.jpg`
- Durations and paths: `/data2/yangky/test/whole_body_tracking/review_videos/t800_reference_audit_20260902/basic_motion_candidate_durations.tsv`

## Basic Mimic Coverage

| Requirement | Current Status | Candidate Files | Audit Note |
|---|---|---|---|
| Straight punch | Pass as reference candidate | `punch_cross_left_e3_stageii.pkl`, `punch_cross_right_e4_stageii.pkl` | User visually accepted cross left/right. Use these before the old `new_data_zhiquan_quanji_tracking.npz`, which is rejected. |
| Hook / swing punch | Rendered, pending manual approval | `punch_hook_left_e5_stageii.pkl`, `punch_hook_right_e6_stageii.pkl` | Re-rendered from clearly named AMASS hook sources. Do not use the old short tracking videos as pass evidence. |
| Front / push kick | Rendered, pending manual approval | Preferred: `kick_push_left_g17_stageii.pkl`, `kick_push_right_g18_stageii.pkl`; backup/reject-prone: `kick_front_g3_stageii.pkl` | Old `accad_g3_front_kick_t800_tracking.npz` failed reference audit. The push-kick sources look closer to competition front kick than the old failed file. |
| Roundhouse / spinning kick | Rendered, pending manual approval | `kick_spinning_back_g4_stageii.pkl`, `kick_roundhouse_left_g8_stageii.pkl`, `kick_roundhouse_right_g9_stageii.pkl`, `kick_reverse_spin_cresent_left_g19_stageii.pkl`, `kick_reverse_spin_cresent_right_g20_stageii.pkl` | Old `new_data_540huixuantitui_tracking.npz` failed reference audit. These are replacement candidates and need visual approval before training. |

## Recovery Coverage

| Requirement | Current Status | Candidate Files | Audit Note |
|---|---|---|---|
| Supine recovery to boxing ready | Full-chain candidates rendered, pending manual approval | `recovery_supine_male1_to_ready.pkl`, `recovery_supine_male2_to_ready.pkl` | The earlier `getup_male1/male2_supine_to_crouch` videos were complete source clips, but the source only reaches crouch. The new yaw-aligned chains append `crouch_to_ready`. |
| Prone recovery to boxing ready | Full-chain candidate rendered, pending manual approval | `recovery_prone_male1_to_ready.pkl` | Built by reversing a crouch-to-prone fragment, then appending `crouch_to_ready`. Useful as a candidate, but dynamic realism must be checked before training. |
| Extra supine candidates | Rendered, lower priority | `recovery_supine_female1_to_ready.pkl`, `recovery_supine_male1_reverse_fall_to_ready.pkl`, `recovery_supine_male2_reverse_fall_to_ready.pkl` | Keep as fallback/augmentation only after the primary supine candidates pass visual audit. |

## Rejected / Not For Training

- `/data2/yangky/test/whole_body_tracking/review_videos/t800_reference_audit_20260902/punching`: rejected for qualification basic mimic. MotionDecode `Punching_Techniques` is too generic and does not clearly match official straight punch/hook primitives.
- `new_data_zhiquan_quanji_tracking.npz`: rejected, reference collapses/prones.
- `new_data_540huixuantitui_tracking.npz`: rejected, reference collapses/prones.
- `accad_g3_front_kick_t800_tracking.npz`: rejected from earlier audit because the robot flips/penetrates.

## Training Gate

Do not start IsaacLab training from any rejected file. The next trainable set should be assembled only from:

- User-approved `official_mimic_candidates` pkl files.
- User-approved `recovery_chains_yaw_aligned` pkl files.
- After approval, convert pkl to tracking npz at 50 Hz, apply stance padding where needed, render the resulting tracking npz again, then start sys01 training.
