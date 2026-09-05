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
| Supine, face up | `pd_stand_y`, `LB+Y` | `supine_to_stance` | Private supine and general recovery runners/assets are present | Public artifacts staged but deliberately unreachable |

The current custom executor therefore must not interpret `LB+X` or `LB+Y` as
"stand up." They only command the corresponding three-second PD pose.

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

No equivalent accepted result exists for prone recovery in this repository.
The local `recovery_prone_male1_to_ready` media is a candidate created by
reversing a fall fragment and appending a ready transition. It has not been
promoted to an accepted policy and must not be deployed as a real-robot action.

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

Keep both recovery actions unreachable in the current real-robot graph until
all of the following have passed:

- a separate recovery-lab FSM, with no combat actions and `LB+RB` fallback;
- simulation of the exact PD-preparation-to-recovery chain, not only the
  official lie-down-to-recovery chain;
- initial joint-error, IMU orientation, finite-output, torque, joint-limit, and
  timeout guards;
- automatic zero-command transition to `walk`, then to boxing-ready only after
  stable base-height/orientation checks;
- first hardware execution while suspended over mats, with an emergency-stop
  operator and exclusion zone.

The currently running custom controller remains unchanged: both PD preparation
poses are available, while no high-dynamic recovery action is reachable.

## References

- [EngineAI Native SDK `urkl_exams` branch](https://github.com/engineai-robotics/engineai_robotics_native_sdk/tree/urkl_exams)
- [Public T800 task graph](https://github.com/engineai-robotics/engineai_robotics_native_sdk/blob/urkl_exams/assets/config/t800/task_motion/default.yaml)
- [Public supine recovery configuration](https://github.com/engineai-robotics/engineai_robotics_native_sdk/blob/urkl_exams/assets/config/t800/rl_supine_to_stance/default.yaml)
- [EngineAI T800 developer documentation](https://engineai.com.cn/open/docs/t800-developer/03?product=t800)
- [URKL competition rules](https://www.engineai.com.cn/tournament-rule-detailed.html)

