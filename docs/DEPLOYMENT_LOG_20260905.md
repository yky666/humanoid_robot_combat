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

## State at Handoff

At the end of the recorded observation window:

- the custom executor was running;
- `robotics.service` was intentionally inactive;
- no motion command had been sent;
- rollback remained available through the saved process group and vendor service.

Runtime logs remain on the controller and are deliberately excluded from Git and
the static deployment manifest.
