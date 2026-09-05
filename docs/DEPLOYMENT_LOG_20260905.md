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
