# URKL Spinning-Kick Import Plan

## Requirements

- Treat `URKL_work.zip` as an imported baseline and preserve its source bytes.
- Do not modify or move the existing GMR, GVHMR, or whole-body-control repos.
- Run the candidate in an isolated simulation setup and render an H.264 video.
- Use the same formal gate: 64 environments x 5 batches = 320 rollouts and
  success rate >= 0.95.
- Promote only a physically passed report into the canonical `best/` state.
- If the imported policy fails, retain it as a baseline and continue the closest
  compatible approved-motion checkpoint rather than restarting from scratch.

## Route

1. Wait for the ZIP upload to finish and verify its central directory and SHA-256.
2. Extract into `baselines/imported/urkl_work/` without touching source repos.
3. Identify robot model, joint order, policy format, observation/action contract,
   motion reference, training framework, and upstream revision.
4. Compare the contract against `t800_policy_v1` and choose the matching simulator.
5. Run one bounded smoke rollout and render a review video.
6. If smoke is structurally valid, run the formal 320-rollout evaluator.
7. Promote on pass; otherwise write a failure report and launch a compatible
   continuation run on GPU1.

## Outputs

- Imported baseline: `baselines/imported/urkl_work/`
- Audit and metrics: `results/roundhouse_540_model33778/`
- Rendered video: `results/roundhouse_540_model33778/video/`
- Cleanup inventory: `cleanup/`

## Outcome (2026-09-05)

- The imported model33778 policy runs through the complete 1633-frame motion in
  isolated MuJoCo and has a fresh 1280x720 H.264 review render.
- Its legacy 140-dimensional observation order and normalization were reproduced
  in the current evaluator. The legacy motion body arrays were reindexed to the
  exact 30-body IsaacLab runtime order before the physical gate.
- Formal result: 47/320 successes (14.6875%), with 271 first failures from
  `ee_body_pos` and four from `anchor_pos`. It was not promoted to `best/`.
- The approved canonical spinning-kick policy is substantially closer to the
  gate, so training resumed from r9 `model_44991.pt` as r10 on physical GPU1.

## Main Risks

- Incomplete or corrupt uploaded archive.
- Policy trained for a different T800 joint/observation order.
- Missing environment code or motion reference.
- A visually plausible policy that fails the formal physical gate.
