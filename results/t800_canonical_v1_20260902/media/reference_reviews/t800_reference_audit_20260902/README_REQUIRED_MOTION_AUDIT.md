# T800 Required Motion Reference Audit - 2026-09-02

## Output

- Punching review videos: `/data2/yangky/test/whole_body_tracking/review_videos/t800_reference_audit_20260902/punching`
- Required retargeted review videos: `/data2/yangky/test/whole_body_tracking/review_videos/t800_reference_audit_20260902/required_retargeted`
- Contact sheets: `/data2/yangky/test/whole_body_tracking/review_videos/t800_reference_audit_20260902/contact_sheets`
- Encoding report: `/data2/yangky/test/whole_body_tracking/review_videos/t800_reference_audit_20260902/video_encoding_check.tsv`
- Decode report: `/data2/yangky/test/whole_body_tracking/review_videos/t800_reference_audit_20260902/video_decode_check.tsv`
- Numeric review hints: `/data2/yangky/test/whole_body_tracking/review_videos/t800_reference_audit_20260902/motion_numeric_review_hints.tsv`

All 57 videos passed H264/yuv420p metadata and ffmpeg decode checks.

## Rendered Candidates

Punching set: 48 videos.

- 4 existing ACCAD T800 tracking motions:
  - `accad_e1_jab_left_t800_tracking.npz`
  - `accad_e2_jab_right_t800_tracking.npz`
  - `accad_e5_hook_left_t800_tracking.npz`
  - `accad_e6_hook_right_t800_tracking.npz`
- 44 MotionDecode `4.Martial_Arts/Punching_Techniques` T800 tracking motions.

Required retargeted set: 9 videos and 9 GMR pickle files.

- Straight punch / cross:
  - `accad_e3_cross_left_stageii.pkl`
  - `accad_e4_cross_right_stageii.pkl`
- Fall/getup fragments:
  - `fall_male1_crouch_to_prone_stageii.pkl`
  - `fall_male1_crouch_to_supine_stageii.pkl`
  - `fall_male2_crouch_to_supine_stageii.pkl`
  - `getup_female1_supine_to_crouch_stageii.pkl`
  - `getup_male1_supine_to_crouch_stageii.pkl`
  - `getup_male2_supine_to_crouch_stageii.pkl`
  - `transition_male2_crouch_to_ready_stageii.pkl`

## Initial Notes

- Existing `new_data_zhiquan_quanji_tracking.npz` and `new_data_540huixuantitui_tracking.npz` remain excluded from training because their reference playback collapses/prones incorrectly.
- ACCAD E3/E4 cross retargets look like better straight-punch candidates than the bad `new_data_zhiquan` file.
- Existing ACCAD jab/hook clips are short, roughly 0.64-0.82 s. Before stable IsaacLab training, they should be padded with stance before/after the strike or mixed with a stance/locomotion policy.
- Existing ACCAD jab/hook tracking arrays have negative `body_pos_w` minima. Visually they do not collapse, but this should be height-checked or regenerated before training.
- MotionDecode punching motions are useful as augmentation, but many are moving/turning/combo motions rather than official straight-punch/hook primitives.
- Current getup data is fragmentary: supine/prone/crouch motions reach crouch or ready, not a complete prone/supine-to-stand recovery in one clip. Training should use either a phased recovery chain or a better full getup source if available.

## Next Gate

Only motions marked visually acceptable after review should be converted into tracking NPZ and added to the IsaacLab training queue on sys01. The likely first training candidates are:

- `accad_e3_cross_left_stageii.pkl`
- `accad_e4_cross_right_stageii.pkl`
- visually approved ACCAD jab/hook, after padding/height check
- visually approved fall/getup fragments, after composing a complete recovery chain
