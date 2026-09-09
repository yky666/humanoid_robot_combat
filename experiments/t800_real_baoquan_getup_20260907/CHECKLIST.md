# Checklist

- [x] Confirm available logs and approximate 500 Hz sampling rate.
- [x] Confirm SDK joint names differ from MuJoCo/policy names for right arm and head.
- [x] Add reusable extraction and rendering tools.
- [x] Generate measured boxing-ready pose report.
- [x] Render `pdstand2baoquan` MuJoCo review video.
- [x] Review/approve the rendered guard pose.
- [x] Wire measured terminal pose into direct-RL get-up task.
- [x] Run short prone/supine simulation smoke checks.
- [x] Extract stable tail hold as the baoquan reference-action asset.
- [x] Start prone/supine longer direct-RL training with the measured target.
- [x] Fix direct get-up evaluation so timeouts are not counted as early failures.
- [x] Add shaped direct get-up tasks for cold-start RL from prone/supine poses.
- [x] Run 500-iteration shaped prone/supine pilots.
- [x] Export playback videos, ONNX policies, and rollout reports for shaped pilots.
- [ ] Pass the 320-rollout success gate.
- [ ] Promote any direct get-up policy to real-robot deployment.
