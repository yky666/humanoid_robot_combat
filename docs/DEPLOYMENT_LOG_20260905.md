# T800 Deployment Log: 2026-09-05

## Scope

This record covers the first controlled startup of the independent T800 Native
SDK package. The robot was reported secured on a load-rated support with an
operator at the emergency stop. No policy motion or gamepad combination was
triggered during this session.

## Package Identity

| Item | Value |
| --- | --- |
| Controller directory | `/home/user/projects/engineai_robotics_qualifier_20260905` |
| SDK upstream commit | `335c60e88772c26c7852d0abd6b3c7439037dd8f` |
| Executor SHA-256 | `ac84559723287c62e40945223fa15b3d0b1ef0e4a1411ee403d9309a8a897670` |
| Runner SHA-256 | `bd3043a029e8daa18d2726cd4f2b0badbf15f4d8038f487f622abee4c7731402` |
| MNN SHA-256 | `bf93f18d0197099a66a1515118612b9093605753aad53a60701dcef53987f063` |
| Manifest | 2,441 files verified |
| Vendor executor SHA-256 | `b2b15834fc404321b88a621a6ff0e1212e59d27a2609ce894f95d88bc87deff2` |

## Controlled Attempts

1. A non-root launch loaded the configuration and initialized the IMU, then
   exited at motor initialization because `/dev/ttyJS` and the motor interface
   require the same root identity used by the vendor service. The vendor service
   was restored and verified active.
2. A root launch using `sudo -E` exited before initialization because sudo
   filtered `LD_LIBRARY_PATH`. The vendor service was restored and verified
   active.
3. The first root launcher exited while sourcing ROS because nounset was enabled
   before `setup.bash`. The vendor service was restored and verified active.
4. The corrected root launcher started successfully at `2026-09-05 10:41:47
   +08:00`.

Every unsuccessful attempt terminated before a policy motion was entered. Only
one `src_executor` existed at any point.

## Runtime Fixes

- `src/executor/main.cc` now resolves the Crashpad handler from
  `ENGINEAI_ROBOTICS_THIRD_PARTY`, with the original `/opt` path as fallback.
- `run_custom_robot_root.sh` sets all runtime paths inside the root shell, checks
  for an existing executor, sources ROS before enabling nounset, and then uses
  `exec` so the saved process group belongs to the controller.

## Successful Startup

The successful process was:

```text
PID/PGID/SID: 10546/10546/10546
User:         root
Executable:   /home/user/projects/engineai_robotics_qualifier_20260905/_install/bin/src_executor
Log:          runtime_logs/custom_root_20260905_104146.log
```

Observed successful initialization:

- configuration loaded from the independent T800 directory;
- RC02 opened on `/dev/ttyJS` at 2,000,000 baud and reported version 9;
- IMU opened on `/dev/ttyACM0` at 460,800 baud;
- MotorRunner and the 25-joint transform initialized;
- ROS 2 bridge and hardware publishers initialized;
- all restricted motion states registered;
- zero fatal/error matches during the first 60 seconds.

The only warning in the observation window was the idle runner reporting a null
parameter and resetting output to zero. The controller remained at motion code
0 during the captured startup log.

## Extended Observation

At `10:44:30.503 +08:00`, the RC02 parser emitted one burst of 24
`skipping unexpected byte` warnings. The last log modification time, file size,
and warning count did not change during subsequent checks. The executor stayed
alive, the hardware and motion ROS 2 topics remained present, and the fatal/error
count remained zero. Treat any renewed or sustained RC02 framing warnings as a
reason to stop and inspect the serial link before sending a motion command.

## State at Handoff

At the end of the recorded observation window:

- the custom executor was running;
- `robotics.service` was intentionally inactive;
- no motion command had been sent;
- the executor had remained active for more than seven minutes;
- rollback remained available through the saved process group and vendor service.

Runtime logs remain on the controller and are deliberately excluded from Git and
the static deployment manifest.

## PD-Prep Integration Follow-Up

EngineAI's official `pd_stand_x` and `pd_stand_y` competition preparation poses
were integrated from `urkl_exams@0d759376cba552b480f267042d5d069ad5d96b50`.
The updated package was staged independently at:

```text
/home/user/projects/engineai_robotics_qualifier_20260905_pdprep
```

The candidate was not started. Its 2,444-file static manifest passed on Nezha,
the executor and custom runner remained ARM64 binaries, and `ldd` reported zero
missing libraries. The final manifest-file SHA-256 is
`29d44a6867ccbdd88c3e19a2fadc1b3068dc17f6a9a23c4ccc639476039278e2`.

During this follow-up, the still-running original executor logged two rejected
requests from `idle` to `pd_stand` at 11:02:08 and 11:02:09. It then reported a
power warning and all motors offline. A second 24-byte RC02 framing burst
occurred at 11:04:58, followed by RC02 receive timeouts at 11:05:31 and
11:05:34.

The original custom process group was stopped at 11:08:30. Verification showed
zero `src_executor` processes, while `robotics.service` was deliberately left
inactive. Inspect RC02 cabling/data integrity, motor power, and emergency-stop
state before starting either the vendor controller or the staged candidate.

## IMU Upgrade And Input Diagnostics

The confidential `engineai-imu-update_1.0.3_2k_arm64.deb` was found on both ARM
hosts with matching SHA-256. It was installed only on Nezha (`163`), where the
YESENSE IMU is physically present; it was not installed on Orin (`162`). The
package upgraded the IMU from `V01.02.06a.1K` to `V01.02.06b`, and the package
status is `engineai-imu-update 1.0.3`. Neither the package nor firmware payload
is committed or copied into the deployment bundle.

After reboot, the vendor controller initialized the new IMU successfully. The
EtherCAT slave reached OP, but the monitor reported `Power board enable status
false`; the controller then repeatedly reported `Motor bus not ready` and all
motors offline. Motion input remains blocked pending physical emergency-stop
and joint-power activation.

A guarded ARM64 keyboard publisher was built on Orin and staged under
`tools/virtual_gamepad/` in the `_pdprep` package. It requires a live custom
`task_state`, an explicit `--arm`, and a state-graph-valid transition. Its test
against the running vendor stack correctly refused input because no custom
`task_state` was present. The final candidate manifest contains 2,447 files;
its manifest-file SHA-256 is
`3685903d0055987ff4a6617c07069be5edf45a8f76bca4d2d1dedecfb4995e01`.

## Initial Hardware Motion Smoke Test

At 11:54:55 the power board changed to enabled, and at 11:54:57 the monitor
reported motors 0 through 24 online. The last `Motor bus not ready` warning was
at 11:54:59. The vendor service was then stopped, its executor was confirmed
absent, and the `_pdprep` package was started as the sole controller:

```text
PID/PGID/SID: 2862/2862/2862
Executable:   /home/user/projects/engineai_robotics_qualifier_20260905_pdprep/_install/bin/src_executor
Log:          runtime_logs/custom_pdprep_20260905_115733.log
```

The custom executor initialized RC02 version 9, virtual-gamepad LCM, the
upgraded IMU, motor runner, 25-joint transform, and ROS 2 without severe, RC02,
power, or motor-readiness warnings. The keyboard publisher first verified the
live `idle` task state without arming, then performed:

1. `idle -> passive`
2. `passive -> pd_stand`
3. `pd_stand -> qualifier_jab_left -> pd_stand`
4. `pd_stand -> passive`

The left-jab runner loaded `obs_dim=140`, `actions=25`, and 165 reference
frames. Its initial-pose error was `0.876165`; the configured guard accepted the
pose, the trajectory finished normally after about 6.3 seconds including the
startup blend, and the configured automatic return entered `pd_stand`.

At final verification, PID 2862 remained alive in `passive`,
`robotics.service` was inactive, and the log contained zero severe entries,
zero RC02 framing/timeout warnings, and zero power or motor-readiness warnings.

At 12:02:50 the same guarded sequence was repeated from `passive`. The second
left-jab trajectory finished normally at 12:03:02, returned automatically to
`pd_stand`, and was placed back in `passive` at 12:03:08. A post-run audit again
found zero severe entries, zero RC02 framing/timeout warnings, and zero power or
motor-readiness warnings. No kick or recovery policy was requested.

## Policy Provenance Correction

Visual feedback from the operator prompted an artifact-level audit after the
two smoke cycles. The trajectory was the accepted left-jab reference (SHA-256
`726e473885fed51dc55c056d074dbaf4a90662f31042181d2487e8eaa724a2c5`),
but the active MNN actor matched the older multi-motion run
`2026-09-02_12-44-27_approved_qualifier_6_t800_joint_v1_from31000`. It did not
match the final accepted jab export from
`2026-09-04_21-53-19_canonical_v1_jab_left_r1/model_4999.pt`.

The earlier cycles remain valid motor, communications, transition, and runtime
smoke tests, but they are invalid as evidence that the final jab policy matches
simulation. The 3-second startup interpolation also explains the deliberately
slow approach to reference frame zero; it does not explain the actor mismatch.
The old PID 2862 was stopped, and the vendor service remained inactive.

Four accepted per-motion ONNX actors were extracted and converted separately
with MNN 2.9.5. Each passed five deterministic ONNX-vs-MNN comparisons at a
`0.001` threshold. The corrected runtime hashes are recorded in the policy
README under the deployment overlay. The failed spinning kick and the old
shared-policy recovery were removed from the reachable task graph. EngineAI's
accepted `rl_supine_to_stance` artifacts remain archived but are not exposed
until their qualified `walk` return path is integrated and tested.

The parameter audit also found that eight shoulder/elbow-pitch gains had been
staged as `Kp=50`, while all four accepted ONNX exports record `Kp=40` for
those joints. No hardware-tuning evidence justified that difference. The
corrected YAML restores the exported values; damping, default joint position,
action scale, observation order, and observation history already matched.
The ONNX robot description leaves identifier gaps before the right arm and
head, whereas the Native SDK uses contiguous identifiers. A semantic-name
audit confirmed that the ordered 25-joint contract is otherwise identical;
the deployment YAML correctly uses the Native SDK identifiers.

The corrected package was staged independently at:

```text
/home/user/projects/engineai_robotics_qualifier_20260905_per_motion
```

Its final 2,451-file manifest passed with manifest-file SHA-256
`0aa7b4b0ce724d576fc80175ea841395a78de0c1bdd27833384c0403dd24d7c8`.
At 12:34:07 it started as PID 6390. A dry-run state subscriber confirmed that
the corrected graph remains in `idle`; no transition or action was sent. The
vendor service was inactive and startup logs contained zero severe entries,
zero RC02 framing/timeout warnings, and zero power or motor-readiness warnings.
The live log is
`runtime_logs/custom_per_motion_params_20260905_123407.log`.

## Correct Package Restart, Walking, And Audio Feedback

After a Nezha reboot, the enabled vendor service started automatically. At
16:15 the operator stopped it and launched the older
`engineai_robotics_qualifier_20260905` package. That package remained in
`idle`; its log confirmed that `pd_stand` and front-kick requests were received
but rejected because neither transition is valid directly from `idle`. The
older package also contained the obsolete shared actor and reachable rejected
motions, so PID 7106 was stopped without executing an action.

The accepted `_per_motion` package was restarted as PID 9985 in `idle`, then
stopped after the walking/audio extension was built. The extension reuses the
unchanged official T800 walking policy (SHA-256
`cbcb90f86dbb2fde39bdc5a25c8d0530d5c79c7a8f84b1f90863d8c9065b6427`)
under `rl_walking_example_runner` and assigns `RB+X`, which was vacated by the
rejected spinning kick. Recovery actions remain unreachable.

Nezha has no ALSA sound card. An Orin listener was therefore installed as
`t800-audio-feedback.service`, using its USB Audio DAC. The controller sends
non-blocking UDP `KEY` and `STATE` events to `192.168.0.162:45800`; one tone
acknowledges a recognized key combination and two tones confirm a state change.
A standalone test packet and the controller's startup `STATE idle` event were
both received without playback errors.

The ARM64 build completed on Orin. Runtime SHA-256 values were
`ac84559723287c62e40945223fa15b3d0b1ef0e4a1411ee403d9309a8a897670`
for `src_executor`,
`4001f60490a001802345fed9ad3362032e92d0825cf822f9ad25f8dc1f131049`
for the input-arbiter library, and
`6fbf7252aa02813c227ab0671079156159c61840284ef223b5915a8ed3672135`
for the keyboard publisher. The Orin audio unit was enabled and active as PID
86729.

The extension was staged independently at:

```text
/home/user/projects/engineai_robotics_qualifier_20260905_walking_audio
```

Its 2,455-file manifest passed with manifest-file SHA-256
`25c58e58b7cd6151fcfe0c4c992031a7ed0758d12b1619d8ca1077a186a5dcec`.
At 16:37:23 it started as PID 12233 and remained in `idle`; the vendor service
was inactive. Its log is `runtime_logs/custom_20260905_163723.log`. After more
than three minutes of observation, severe, power, motor-offline, motor-bus, and
RC02 timeout/framing fault counts remained zero. Walking and combat actions
were not triggered during this deployment check.

## Boxing-Ready And Runtime Policy Audit

A later operator test entered the corrected straight-punch policy. At
16:53:08 its automatic return to `pd_stand` was rejected repeatedly because
`J13_SHOULDER_PITCH_L` differed from the stand target by `1.43219 rad`, above
the configured `1.2 rad` guard. The operator selected `passive`, then re-entered
`pd_stand` successfully. At 16:54:04 the official walking policy entered and
remained active until a deliberate return to `pd_stand` at 16:54:33. A direct
walking-to-punch request was correctly rejected by the restricted graph.

Read-only inspection of the untouched vendor package found private
`rl_mimic_boxing` and `rl_mimic_boxing_classic` runners with dedicated
`switch_to_boxing_idle` and `switch_to_stance_idle` paths. Their packed assets
are stored in `/apps/engineai_robotics/assets/config/t800.bson`. Resolving the
vendor boxing plugin against the custom runtime produced an undefined
`data::RlMimicBoxingParam` symbol, so the private runner is not ABI-compatible
as a drop-in plugin and was not copied or loaded.

The four reachable per-motion MNNs were independently compared again with
their archived accepted ONNX exports using five deterministic inputs each. All
20 comparisons passed at the `0.001` threshold. The four MNN hashes on Nezha
matched the repository overlay. A metadata audit also passed the observation,
action, gains, scaling, history, and semantic 25-joint mapping for every active
motion. Full results and the recommended boxing-ready integration route are in
`docs/T800_BOXING_READY_AND_POLICY_AUDIT.md`.

## Fall-Recovery Audit

The public `urkl_exams` branch was fetched again at `0d75937`. Its `LB+X`
`pd_stand_x` and `LB+Y` `pd_stand_y` states are static three-second PD
preparation poses, not recovery actions. The branch publishes a complete
`supine_to_stance` MNN-plus-reference-trajectory action but no corresponding
prone action. Its FSM invokes back recovery separately from `passive` with
`START+D-pad up` and automatically enters `walk` when the trajectory ends.

The archived public supine action passed 320/320 independent fresh-process
Native SDK MuJoCo trials with bounded velocity perturbations and a median
recovery-to-walk time of approximately 3.05 seconds. Those trials used
`stance_to_supine -> passive -> supine_to_stance -> walk`, not the competition
`pd_stand_y` preparation pose. The latter differs from the selected recovery
reference start by as much as 2.32 rad on one joint, so the 0.3-second runtime
blend is not sufficient evidence for a direct hardware trial.

Read-only inspection of the untouched vendor release found private prone,
supine, and general fall-recovery assets and dedicated recovery runners in the
packed release. The vendor therefore has both directions, but its private
runner ABI and protected assets were not copied into the custom deployment or
the public repository. PID 12233 and the custom graph were left unchanged; no
recovery or other motion command was sent. The full findings and integration
gates are in `docs/T800_FALL_RECOVERY_AUDIT.md`.

A subsequent read-only check confirmed that `rl_supine_to_stance` is mapped in
the deployed parameter registry but absent from the active task graph and both
input maps. It is therefore not controllable in the current custom executor.
The last logged state at 18:20:05 was `pd_stand_y`; repeated earlier requests
from floor poses to upright `pd_stand` were rejected by its 1.2-rad joint-bias
guard. Direct pose comparison gives maximum differences of 2.185 rad from
`pd_stand_y` and 2.590 rad from `pd_stand_x` to upright stand. The guard remains
enabled because the upright PD runner assumes both feet, and no arms, are in
contact. No configuration or live process was changed.

## Supine Recovery State Staging

The official public `supine_to_stance` task was added to a new independent
deployment candidate. It is reachable only from `passive` with the official
`START+D-pad up` binding, permits `LB+RB` interruption, and automatically
returns to zero-command `walk`. The SSH keyboard publisher adds `u`; the Orin
audio service was updated and restarted successfully for request/state tones.

The recovery trajectory frame 90 is close to upright `pd_stand` in joint space
(maximum difference 0.687 rad), confirming the operator's recollection. Its
base is nevertheless supine at approximately 0.14 m height. The official
`stance_to_supine` terminal pose is a better match (maximum joint difference
0.362 rad), while `pd_stand_y` differs by 2.320 rad. The global 1.2-rad upright
PD bias guard was not weakened or bypassed.

The candidate was staged at:

```text
/home/user/projects/engineai_robotics_qualifier_20260905_recovery
```

Static bundle validation passed. The rebuilt keyboard tool is ARM64, resolves
its packaged LCM library, and completed a no-command state subscription. Its
SHA-256 is `f35b8a829e1197cc50dd70609edb2a00606cb7ad219e9733ef263a9c271fa610`;
the 2,455-file manifest passed with manifest SHA-256
`a218ebff59a49293f0a5179b910e5e970b51365907af6d86ea9ab3afadadffa6`.
The existing PID 12233 remained on the `_walking_audio` package, so the new
state is staged but not active. No motion command was sent.

## Motor-Bus Incident And Force-PD Lab

After a controller restart, the `_recovery` executor started as PID 2071 while
all motors were offline. The EtherCAT master and its `foe` slave were in `OP`,
but the SDK repeatedly reported `Motor bus not ready (self-check)`. A recovery
request was accepted at the FSM level while the bus was unavailable and
produced out-of-range policy targets. After motor communication returned, a
second recovery run still produced targets above 8 rad before automatically
entering `walk`. Further public supine-recovery trials were stopped.

At 21:51:34 the motor monitor cleared its offline state. Subsequent
`passive -> pd_stand` requests were correctly rejected by the 1.2-rad guard;
the largest observed mismatch was about 3.18 rad. Recovery requests made from
`walk` or `pd_stand_x` were also correctly rejected by the state graph. The
executor was returned to `passive` through the guarded virtual gamepad.

An opt-in package was then staged at:

```text
/home/user/projects/engineai_robotics_qualifier_20260905_force_pd_lab
```

It contains only the five idle/passive/PD states and uses the SDK's existing
`strict_motion_check: false` plus held-LT force confirmation. Its 2,458-file
manifest passed with SHA-256
`e2d1a37865b6582fc10fc3dbb91916f6d5006bdcb77ed9da0b9fe73683c7b99f`.
It was staged but not activated because replacing the root-owned PID 2071
requires a deliberate local sudo operation. No force-PD motion was triggered.

## 22:19 Controller And FSM Reconciliation

A read-only status check found `robotics.service` inactive and no custom
`src_executor` process. EtherCAT master 0 remained active in `Operation`; the
single `foe` slave was `OP`, the Ethernet link was up, and the master reported
zero lost frames. This is a healthy transport reading, not proof that the
downstream motors are enabled, because no SDK motor runner was alive to publish
motor readiness.

The recovery PID 2071 had stopped. A later attempt at 22:16:45 to start the
`_force_pd_lab` copy reached resident-task initialization, then exited with
`Catch Exception: bad optional access`. Inspection also found that this staged
copy still selected `task_motion/qualifier_robot`, despite the separate
`force_pd_lab.yaml` file. It therefore was not an isolated five-state graph and
must not be restarted under the lab name until the selected mode scopes and
startup behavior are corrected.

The primary `_recovery` graph itself remains intact: it contains `idle`,
`passive`, `pd_stand`, official walking, `pd_stand_x`, `pd_stand_y`, the public
`supine_to_stance` definition, and the four accepted combat policies. These are
separate control paths, not one serial chain. Upright operation uses
`passive -> pd_stand -> walk/combat`; supine recovery uses
`pd_stand_y -> passive -> supine_to_stance -> walk`. The open SDK has no public
prone stand-up action after `pd_stand_x -> passive`. An upright `pd_stand`
insertion while the robot is on the floor is rejected as an unsafe contact-model
mismatch, not treated as a missing FSM edge.

The independent lab copy was then corrected without starting an executor. Its
robot-mode scopes now select `task_motion/force_pd_lab` and
`global_options/force_pd_lab`; the lab graph was aligned with the SDK's
`cpu + tasks` schema. Static validation passed, and the regenerated 2,458-file
manifest passed with manifest SHA-256
`59de372eb97c9eabc758bee708045cd4686c64c503102283403731c832095097`.
A final 22:29 status check still showed both controllers stopped and the
EtherCAT `foe` slave in `OP`. No motion command was sent.

The primary graph was then hardened while stopped. The
`supine_to_stance` task, official policy, trajectory, key metadata, and
automatic `walk` return remain in the package, but its incoming transition from
`passive` was removed. This makes live requests fail closed while preserving
the integration for guard implementation and renewed qualification. Walking,
both PD preparation poses, and all four accepted combat actions remain
reachable through their existing edges.
