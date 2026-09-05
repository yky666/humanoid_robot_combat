# Imported Model33778 Evaluation

## Candidate

- Bundle: `URKL_work.zip`
- Policy: `roundhouse_540_midpush_model33778_2000iter`
- Motion: 1633 frames at 50 Hz
- Legacy actor: 140 observations, 25 actions, embedded empirical normalization

## Simulation

The candidate executed the complete reference in an isolated MuJoCo environment
and remained standing at the end. This confirms that the SDK asset is runnable,
but it is a visual/sim2sim result rather than policy acceptance.

Review video:

`video/model33778_mujoco_fresh_20260905.mp4`

- Codec: H.264
- Resolution: 1280x720
- Frame rate: 50 fps
- Duration: 32.66 seconds
- SHA-256: `0cc78be62f96505d6220f822d1670dafa688e04a7bbcc30d2da0a787a1149022`

## IsaacLab Gate

The first attempted gate used the legacy depth-first body columns directly and
was invalid because IsaacLab indexes the arrays in articulation runtime order.
`scripts/reindex_t800_motion_bodies.py --target runtime30` produced the corrected
30-body reference. Initial end-effector error then dropped from about 1.35 m to
at most 5.46 cm across the five formal batches.

Corrected formal report:

`formal_320_runtime30/physical_report.json`

- Rollouts: 64 environments x 5 batches = 320
- Successes: 47/320 (14.6875%)
- Required: at least 304/320 (95%)
- First failures: 271 `ee_body_pos`, 4 `anchor_pos`
- Decision: rejected; no canonical `best/` marker or SDK staging was created

The imported checkpoint and render remain a useful baseline. Continuation of the
main project uses the already-audited canonical spinning-kick r9 checkpoint,
which was much closer to the stability gate, and resumes it as r10 on GPU1.

## Single-Kick Branch

Kinematic inspection found three right-foot kick peaks at source frames 391,
936, and 1374. The middle cycle has the best standalone boundary conditions:
frames 849 through 1297 return to nearly the same pose, with 0.112 rad endpoint
joint distance and 0.147 m net root displacement.

`roundhouse_540_single_middle_f0849_f1297_hold20_runtime30.npz` contains that
single cycle, rebased to zero initial XY and heading. Twenty zero-velocity frames
(0.4 seconds) were added at each end, producing 489 frames / 9.78 seconds. Its
joint order and finite-value checks pass `t800_policy_v1`.

The imported policy visibly executes one airborne kick in the single-cycle
MuJoCo render. This is only a prepared continuation candidate; it has not yet
run the 320-rollout IsaacLab gate and is not accepted or staged to the SDK.
