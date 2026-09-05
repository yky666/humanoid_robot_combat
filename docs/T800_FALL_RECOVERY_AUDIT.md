# T800 Fall-Recovery Audit

Audit date: 2026-09-05. This was a read-only source, artifact, and running-system
inspection. No recovery command or other robot motion was triggered.

## Result

`pd_stand_x` and `pd_stand_y` are static PD reset/preparation poses, not
stand-up actions. The public `urkl_exams` branch provides one complete recovery
action, for a robot lying on its back. The installed vendor release contains
both back and front recovery implementations, but the front implementation is
not published in the open Native SDK.

| Fall orientation | Preparation pose | Public stand-up action | Installed vendor release | Current custom graph |
| --- | --- | --- | --- | --- |
| Prone, face down | `pd_stand_x`, `LB+X` | None | Private prone recovery runner and assets are present | Preparation only; no stand-up action |
| Supine, face up | `pd_stand_y`, `LB+Y` | `supine_to_stance` | Private supine and general recovery runners/assets are present | Exposed in the staged `_recovery` package, not the currently running package |

The current custom executor therefore must not interpret `LB+X` or `LB+Y` as
"stand up." They only command the corresponding three-second PD pose.

## Current Custom Integration

The recovery configuration and public supine MNN/trajectory are present in the
deployment, and `mode.yaml` maps the `rl_supine_to_stance` parameter tag. The
staged `_recovery` package adds the task to
`task_motion/qualifier_robot.yaml`, allows entry only from `passive`,
binds the physical gamepad to `START+D-pad up`, and adds keyboard key `u`.

At the latest read-only check, the vendor service was inactive, the custom
executor was PID 12233, and the last recorded state transition at 18:20:05 was
to `pd_stand_y`. No command was sent during the check.

The staged package is `/home/user/projects/engineai_robotics_qualifier_20260905_recovery`.
It has not replaced the running `_walking_audio` executor and no recovery has
been executed on hardware.

## Public Supine Chain

The public T800 state graph defines this sequence:

```text
pd_stand_y --LB+RB--> passive --START+D-pad up--> supine_to_stance
supine_to_stance --automatic--> walk (zero stick command)
```

The public graph allows `supine_to_stance` only from `passive`. It does not
define a direct `pd_stand_y -> supine_to_stance` transition. The emergency
fallback remains `LB+RB -> passive` from every active state.

The public recovery bundle consists of:

- `rl_supine_to_stance/default.yaml`
- `rl_supine_to_stance/trajectory/T800_supine_to_stance.npy`
- `rl_supine_to_stance/policy/T800_supine_to_stance.mnn`

The trajectory is a 197 x 34 float16 array. The runner selects frames 90 through
180 at 30 Hz and interpolates them to the 100 Hz control loop. It uses
`residual_control: true`: desired joint position is the reference trajectory
plus the clipped MNN policy output, rather than an open-loop trajectory replay.
Entry from the measured joint state is linearly blended for 0.3 seconds. At
trajectory completion, the FSM requests the automatic transition to `walk`.

Unlike `PdStandRunner`, `RlMimicTrajectoryRunner::Enter()` performs no upright
joint-bias rejection. It records the measured joints and uses them as the start
of that 0.3-second blend; torque limiting remains enabled. Entering the official
recovery from `passive` therefore already provides the requested bias-independent
floor path without disabling `strict_motion_check` for upright PD states.

This architecture is consistent with a fast kinematic get-up followed by RL
correction and a locomotion controller holding the final stance. Source and
artifact inspection alone cannot establish that the reference is specifically
the two-leg scissor movement described by an operator.

## Evidence And Limits

The archived public supine bundle has matching SHA-256 values in the SDK tree,
canonical result archive, and staged robot directory:

```text
policy     deb9974b1f4f4a7e77801f8c9c6e77f599caab0ca4dd7709fe0bae55870e0e86
trajectory c2f19c164093701311634024eb27999fed4631a00d38d507f8aa306ee138c161
```

It passed 320/320 independent fresh-process Native SDK MuJoCo trials with
bounded linear and angular velocity perturbations immediately before recovery.
This is a Native SDK MuJoCo result, not an IsaacLab vectorized rollout. The
recorded median recovery-to-`walk` time is approximately 3.05 seconds.

That campaign used the official `stance_to_supine -> passive ->
supine_to_stance -> walk` chain. It did not qualify the competition
`pd_stand_y -> passive -> supine_to_stance` chain. In addition, the public
`pd_stand_y` joint target is not the same as frame 90 of the recovery reference;
the maximum per-joint difference is about 2.32 rad. The runner's 0.3-second
blend reduces discontinuity but does not prove that this alternate entry is
physically safe.

The upright `pd_stand` joint target is much closer to recovery frame 90: its
maximum per-joint difference is 0.687 rad, with no joint over the 1.2-rad stand
guard. This is joint-space similarity only. At frame 90 the reference base is
about 0.14 m above the floor and its quaternion represents a supine body; the
same limb angles in upright `pd_stand` have a different base orientation and
contact state. The official `stance_to_supine` terminal joints are closer
again, with a maximum difference of 0.362 rad.

No equivalent accepted result exists for prone recovery in this repository.
The local `recovery_prone_male1_to_ready` media is a candidate created by
reversing a fall fragment and appending a ready transition. It has not been
promoted to an accepted policy and must not be deployed as a real-robot action.

## Why The PD Bias Guard Must Remain

`PdStandRunner::Enter()` rejects a transition when any measured joint differs
from the target by more than the configured threshold. The real-robot upright
`pd_stand` threshold is 1.2 rad. Only after this check succeeds does the runner
interpolate from measured joints to the upright target over three seconds.

The open runner has a force-start path when `strict_motion_check` is disabled
and the operator holds LT. The deployed release configuration deliberately has
`strict_motion_check: true`, so that override is unavailable. It must not be
enabled for floor recovery.

From the exact competition poses, the largest target difference is 2.185 rad
for `pd_stand_y -> pd_stand` and 2.590 rad for `pd_stand_x -> pd_stand`. More
importantly, the PD runner declares both feet in contact and both arms out of
contact throughout execution. That assumption is false while the torso, arms,
or back are on the floor. Increasing/removing the bias threshold would only
permit a smooth joint-space interpolation under an invalid contact model; it
would not turn PD stand into a dynamically balanced get-up controller.

Do not implement either of these upright-PD shortcuts:

- `pd_stand_y -> pd_stand` with the bias check bypassed;
- `pd_stand_x -> pd_stand` followed by a standing punch policy used against
  the floor.

The second route also applies a policy outside its trained base orientation,
contact set, observation distribution, and terminal state. Arm-ground impact
must be modeled explicitly if it is part of a recovery maneuver.

## Recommended Two-Orientation Design

The public supine policy can be evaluated in an isolated recovery graph using:

```text
pd_stand_y -> passive -> guarded supine_to_stance -> walk(zero command)
             -> stable-upright confirmation -> boxing-ready
```

The prone fallback may reuse the accepted supine get-up only after a dedicated,
contact-aware roll-over action has safely reached its required supine state:

```text
pd_stand_x -> passive -> prone_to_supine
             -> settled-supine confirmation -> supine_to_stance
             -> walk(zero command) -> stable-upright confirmation
             -> boxing-ready
```

`prone_to_supine` may use an arm push/ground strike as one phase, but it must be
trained or authored as a recovery action with arm contacts, torque/impact
limits, base-orientation checks, and a verified terminal supine pose. It must
not invoke the upright `pd_stand` state in the middle of the maneuver.

## Installed Vendor Release

Read-only inspection of `/apps/engineai_robotics` found private prone, supine,
and four-direction fall-recovery configurations and models, plus dedicated
`rl_recover` and `rl_recover_prone` runner libraries. This confirms that the
vendor controller has both orientations available. Those assets are packed in
the protected `t800.bson` release and depend on private runner parameter ABIs.

Do not copy those models into this repository or mix their libraries into the
open custom executor. The supported choices are:

1. Run the complete vendor controller for vendor recovery, using the vendor's
   documented state machine and preserving mutual exclusion with the custom
   executor.
2. Ask EngineAI for the competition-supported prone recovery package/API for
   the open Native SDK.
3. Train and qualify an independent prone policy, including the exact
   `pd_stand_x` entry and a final boxing-ready/locomotion transition.

## Safe Integration Gate

Do not activate the staged supine task or add prone recovery until all of the
following have passed:

- a separate recovery-lab FSM, with no combat actions and `LB+RB` fallback;
- simulation of the exact PD-preparation-to-recovery chain, not only the
  official lie-down-to-recovery chain;
- initial joint-error, IMU orientation, finite-output, torque, joint-limit, and
  timeout guards;
- automatic zero-command transition to `walk`, then to boxing-ready only after
  stable base-height/orientation checks;
- first hardware execution while suspended over mats, with an emergency-stop
  operator and exclusion zone.

The currently running custom controller remains unchanged. The independently
staged package exposes only the accepted public supine recovery; it must be
started deliberately after the running controller is safely returned to
`passive` and stopped.

## References

- [EngineAI Native SDK `urkl_exams` branch](https://github.com/engineai-robotics/engineai_robotics_native_sdk/tree/urkl_exams)
- [Public T800 task graph](https://github.com/engineai-robotics/engineai_robotics_native_sdk/blob/urkl_exams/assets/config/t800/task_motion/default.yaml)
- [Public supine recovery configuration](https://github.com/engineai-robotics/engineai_robotics_native_sdk/blob/urkl_exams/assets/config/t800/rl_supine_to_stance/default.yaml)
- [EngineAI T800 developer documentation](https://engineai.com.cn/open/docs/t800-developer/03?product=t800)
- [URKL competition rules](https://www.engineai.com.cn/tournament-rule-detailed.html)
