# T800 Boxing-Ready and Policy Audit

Audit date: 2026-09-05 (Asia/Shanghai)

## Scope

This audit answers three deployment questions:

1. Whether the vendor boxing-ready state can be reused by the isolated custom controller.
2. Whether every reachable custom action uses a policy that passed the formal 320-rollout gate.
3. Whether the accepted ONNX exports, MNN runtime models, and 25-joint SDK mapping agree.

No robot motion was requested during this audit. The vendor installation under
`/apps/engineai_robotics` was inspected read-only and no vendor file was copied
into this repository.

## Vendor Boxing-Ready State

The installed vendor package is release commit
`b250f38ca4f4b2edf38a112e96de95a520810945` from 2026-07-31. Unlike the public
Native SDK, it contains these private runners:

```text
libsrc_runner_rl_mimic_boxing.so
libsrc_runner_rl_mimic_boxing_classic.so
```

Their symbols and diagnostic strings confirm dedicated
`switch_to_boxing_idle` and `switch_to_stance_idle` paths, profile-specific
trajectory loading, transition-ratio computation, and stability checks. The
packed vendor configuration also names `rl_mimic_boxing/default.yaml`, boxing
idle-switch policies, and idle-switch trajectories. This is the guarded,
slightly crouched pose observed in the vendor application.

The public `origin/urkl_exams` branch does not contain this runner or state. It
only adds `pd_stand_x` and `pd_stand_y`, which are fall-recovery preparation
poses.

### Drop-in compatibility result

The private boxing runner is not a drop-in plugin for the custom executor:

- The vendor executor links both boxing libraries directly; the custom
  executor does not.
- Resolving the vendor boxing library against the custom runtime leaves
  `data::RlMimicBoxingParam` undefined.
- Boxing configuration, policies, and trajectories are packed in the vendor's
  223 MB `assets/config/t800.bson`, while the public/custom runtime uses an
  unpacked YAML asset tree.
- Replacing the custom core/data libraries with vendor versions would mix two
  unverified ABIs and could invalidate the custom runner's safety changes.

Do not copy or preload the vendor boxing `.so` files into the current package.
Do not alternate the vendor and custom executors as a motion-level state
transition; only one executor may control the robot and process handoff does not
provide a continuous motor-command contract.

### Recommended integration

Use the vendor boxing idle as the behavioral reference for a native custom
`boxing_ready` state:

1. In a separately approved, supported hardware session, run the unmodified
   vendor controller and record the settled boxing-idle joint state and command
   gains through the normal telemetry interface.
2. Reproduce and validate that pose in simulation. A static PD version is
   suitable only as a smooth pose bridge; it does not reproduce the vendor
   policy's dynamic balance.
3. Train a closed-loop boxing-ready policy with perturbation recovery, then
   train each action with `boxing_ready -> action -> boxing_ready` references.
4. Add explicit state-graph edges only after the complete entry, hold, action,
   and return sequence passes simulation and suspended-hardware tests.

## Current Transition Smoothing

`pd_stand` uses a three-second quintic interpolation from the measured joint
state to the official fixed stand pose. Each custom action currently performs a
separate three-second linear blend of position, stiffness, and damping from the
previous command to policy/reference frame zero. The trajectory frame counter
is held at zero during that blend.

This provides command continuity but is not sufficient evidence of a stable
transition. There is no state-graph transitioner between the states, and the
action reference frame zero is already moving. The observed straight-punch
return also exceeded the `pd_stand` entry threshold at
`J13_SHOULDER_PITCH_L` (`1.43219 rad` versus `1.2 rad`), causing repeated failed
automatic returns. A boxing-ready middle pose can reduce pose distance, but the
actions must be retrained or supplied with stationary entry/exit segments; a
longer interpolation alone does not fix the policy/reference mismatch.

## Formal Qualification

The four actions reachable in the restricted real-robot graph use independent
accepted policies:

| Motion | Accepted checkpoint | Formal result |
| --- | --- | ---: |
| Front kick | `model_44991.pt` | 320/320 |
| Straight punch | `model_4999.pt` | 320/320 |
| Hook punch | `model_33796.pt` | 320/320 |
| Left jab | `model_4999.pt` | 316/320 |

All four satisfy the configured minimum success rate of 95%. These results
qualify the policies in the IsaacLab evaluation contract; they do not qualify
the current real-robot entry and return transitions. The spinning kick and the
older shared six-motion actor remain unreachable.

## ONNX to MNN Verification

Training exports a multi-output ONNX wrapper with `obs` and `time_step` inputs.
`tools/prepare_qualifier_policy.py` validates the source and extracts only the
actor subgraph:

```text
obs:     float32 [1, 140]
actions: float32 [1, 25]
```

Each accepted per-motion actor was converted with MNN 2.9.5. On 2026-09-05 the
four deployed MNN files were independently rechecked against their archived
accepted ONNX source using five deterministic inputs per motion. All 20
comparisons passed at the `0.001` threshold. The hashes on the Nezha controller
match the repository deployment overlay exactly.

| Motion | Accepted ONNX SHA-256 | Deployed MNN SHA-256 |
| --- | --- | --- |
| Front kick | `5138f26d954d66741e5d84b85c8b7594e6d0f59dfe1b69a857b2d9f7e2f85069` | `f025f857f074cd5073b6f7abb4eedf677af126c9e934c6706542aa911ca6d8f3` |
| Straight punch | `f0aebc1bed7192be0cb861bad7693063fe87ec014d8daf932aecae3b82e2cf0d` | `7a863b258a5700942628a7ced386f67d9adfd9eb236dbe14ed082f7b91a4b1fa` |
| Hook punch | `62f898df5733d9ee31860994cf17ad94c02229f9d4ae2c7164a8aa809925c44f` | `f486411792b0744922e195b18c4f2fff09c7ff9ef119a7789af86a27a9195e4b` |
| Left jab | `7b13114adaca9467c95860a82c98d0171ff122cb3aebbfd8a6ac2796e98c092e` | `a0437216bb8d7d9f339840a804c34ee5c874ab6c193ff1072887aa9a51695697` |

The training model names leave identifier gaps for the right arm and head
(`J20..J28`), while the Native SDK uses contiguous identifiers (`J18..J24`).
The positional semantic order is identical. A fresh metadata audit passed all
four policies for joint order, default positions, stiffness, damping, action
scale, observation order, observation history, and the 140-element observation
dimension.

