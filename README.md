# Humanoid Robot Combat

Reproducible T800 combat-motion workspace built around EngineAI Native SDK,
IsaacLab, Whole Body Tracking/BeyondMimic, and GMR. The repository contains the
working code overlays plus the canonical motions, policies, evaluation reports,
checkpoints, and rendered review media produced through 2026-09-05.

## Qualification Status

Formal policy acceptance uses 64 environments x 5 batches = 320 rollouts and a
minimum success rate of 0.95. TensorBoard tail metrics and visual playback are
not treated as policy acceptance.

| Motion | Result | Formal rollouts |
| --- | --- | ---: |
| Hook punch | Passed | 320/320 |
| Front kick | Passed | 320/320 |
| Straight punch | Passed | 320/320 |
| Left jab | Passed | 316/320 |
| Supine recovery | Passed, official EngineAI MNN | 320/320 |
| Spinning kick | Failed tail gate | No formal rollout |
| Stand/action/stand joint policy | Blocked | Not started |

The latest spinning-kick run (`r9`) passed timeout, anchor, episode-length, and
joint-error checks, but its end-effector termination rate was 4.58% against a
3% maximum. It is archived as a failed experiment and is not in `best/`.

## Layout

```text
GMR/                              Retargeting source snapshot and T800 assets
whole_body_tracking/              IsaacLab tracking, training, evaluation code
engineai_native_sdk_integration/  SDK overlays and binary-capable patches
isaaclab_integration/             Local IsaacLab overlay and patch
manifests/                        Dependency revisions and archive metadata
results/t800_canonical_v1_20260902/
  approved_reference/             Canonical NPZ/PKL inputs and reference videos
  checkpoints/final_rounds/       Last checkpoint from every reported train run
  policies/                       Accepted PT/ONNX policies and run parameters
  qualification/                  Tail gates, 320-rollout reports, best manifests
  recovery_official/              EngineAI supine-to-stance MNN and trajectory
  media/                          All retained T800 review and policy videos
  archives/                       Complete compressed logs and experiment outputs
```

Large generated files are tracked with Git LFS. Run `git lfs pull` after cloning
to materialize policies, motions, videos, and compressed archives.

See [the archive notes](docs/ARCHIVE_20260905.md) and
[the canonical result README](results/t800_canonical_v1_20260902/README.md) for
provenance and restore commands.

## Deliberately Excluded

The archive omits reproducible runtime bulk: Conda environments, Isaac/Omniverse
caches, build trees, WandB caches, and every 100-iteration checkpoint. It keeps
the final checkpoint from each reported training round, all formal reports,
complete compressed canonical logs, and all retained T800 videos. The original
working directories were not reset or modified while creating this snapshot.
