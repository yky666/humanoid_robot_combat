# T800 Real-Robot Deployment

This runbook reconstructs, stages, starts, observes, and rolls back the custom
EngineAI Native SDK controller without overwriting the vendor application.

## Deployment Record

| Item | Value |
| --- | --- |
| Robot | EngineAI T800 |
| Controller | Nezha, ARM64 Ubuntu 22.04, ROS 2 Humble |
| Build host | Jetson Orin, ARM64 Ubuntu 22.04 |
| SDK base | `335c60e88772c26c7852d0abd6b3c7439037dd8f` |
| Corrected candidate | `/home/user/projects/engineai_robotics_qualifier_20260905_per_motion` |
| Vendor package | `/apps/engineai_robotics` |
| MNN runtime | 2.9.5 |
| Policy contract | `obs[1,140] -> actions[1,25]`, float32 |
| Official PD reference | `urkl_exams@0d759376cba552b480f267042d5d069ad5d96b50` |
| IMU firmware | `V01.02.06b`, package `engineai-imu-update 1.0.3` |
| Last controller state | Corrected custom PID `6390` running in `idle`; vendor service inactive |

The package was built on ARM64, checked against the controller's hardware
libraries, transferred over the robot LAN, and verified with a 2,451-file
SHA-256 manifest. The vendor package was not modified.

See the [2026-09-05 deployment log](DEPLOYMENT_LOG_20260905.md) for the first
controlled startup and rollback record.

The `_pdprep` package completed two guarded hardware cycles, but the subsequent
provenance audit invalidated their policy-fidelity result: the jab trajectory
used an older six-motion actor. That executor was stopped. The corrected
`_per_motion` package binds each active motion to its accepted actor and was
started only to persistent `idle`; no corrected action has run on hardware yet.

## Network Topology

```text
remote workstation
  -> Tailscale
  -> Windows 11 workstation (jump host and TCP port proxy)
  -> Ethernet 192.168.0.0/24
       |- Jetson Orin 192.168.0.162:22
       `- Nezha      192.168.0.163:22
```

The Windows Ethernet adapter uses a static address in the same subnet. In the
recorded setup, forwarded TCP ports exposed Orin SSH and Nezha SSH through the
Windows Tailscale address. Do not commit SSH passwords or private keys.

Verify both the listener and the end-to-end SSH path; a successful local
`Test-NetConnection` alone only proves that the Windows listener is reachable.

## Reconstruct the SDK Tree

```bash
git clone https://github.com/engineai-robotics/engineai_robotics_native_sdk.git
cd engineai_robotics_native_sdk
git checkout 335c60e88772c26c7852d0abd6b3c7439037dd8f
cp -a \
  /path/to/humanoid_robot_combat/engineai_native_sdk_integration/deploy_20260904/overlay/. \
  ./
```

The overlay is the authoritative runnable delta because it includes both
modified upstream files and newly added policies, trajectories, configuration,
and tools. The adjacent patch is retained for reviewing upstream-tracked source
changes; it is not a substitute for the new files in the overlay.

Validate the policy bundle before compiling:

```bash
python tools/validate_qualifier_bundle.py
```

## ARM64 Build

The controller's installed third-party package is runtime-oriented and may omit
development tools. The recorded build used read-only copies of the matching
third-party and hardware SDK trees on the Orin build host.

```bash
source /opt/ros/humble/setup.bash

cmake -S . -B build/aarch64 \
  -DBUILD_TYPE=release \
  -DBUILD_TESTS=OFF \
  -DBUILD_ROS2=ON \
  -DBUILD_DCHECK=ON \
  -DROS2_VERSION=humble \
  -DENGINEAI_ROBOTICS_THIRD_PARTY_ROOT=/path/to/matching/engineai_robotics_third_party \
  -DENGINEAI_ROBOTICS_HARDWARE_ROOT=/path/to/matching/engineai_robotics_hardware

cmake --build build/aarch64 -j"$(nproc)"
cmake --install build/aarch64
```

Confirm that both binaries are ARM64:

```bash
file build/aarch64/_install/bin/src_executor
file build/aarch64/_install/lib/libsrc_runner_rl_dance_example.so
```

## Package Layout

The independent runtime directory contains:

```text
engineai_robotics_qualifier_20260905_per_motion/
  _install/                         ARM64 binaries and runtime dependencies
  assets/config/t800/               base T800 and custom motion configuration
  assets/resource/                 T800 model XML, meshes, and environment
  DEPLOYMENT_MANIFEST.sha256       package integrity manifest
  DEPLOYMENT_NOTES.md              package identity and safety boundary
  run_robot.sh                     guarded vendor-compatible launcher
  run_custom_robot_root.sh         root launcher with isolated runtime paths
```

Create a manifest from the package root after final assembly:

```bash
find . -type f ! -path './runtime_logs/*' ! -name DEPLOYMENT_MANIFEST.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum > DEPLOYMENT_MANIFEST.sha256
```

## Preflight

Complete every item before stopping the vendor controller:

1. Secure the robot with a load-rated stand or harness and keep both feet clear.
2. Clear the exclusion zone and place one operator on the tested emergency stop.
3. Verify battery, power, communications, and joint state are normal.
4. Confirm the selected policy's initial reference pose before triggering it.
5. Verify package integrity and dynamic libraries.
6. Record the vendor service state and executable hash for rollback.

Run the static checks on Nezha:

```bash
cd /home/user/projects/engineai_robotics_qualifier_20260905_per_motion
sha256sum -c DEPLOYMENT_MANIFEST.sha256

source /opt/ros/humble/setup.bash
export ENGINEAI_ROBOTICS_DIR="$PWD"
export ENGINEAI_ROBOTICS_THIRD_PARTY="$PWD/_install/engineai_robotics_third_party"
export ENGINEAI_ROBOTICS_HARDWARE=/opt/engineai_robotics_hardware
export LD_LIBRARY_PATH="$ENGINEAI_ROBOTICS_THIRD_PARTY/lib:$ENGINEAI_ROBOTICS_THIRD_PARTY/lib/runtime:$ENGINEAI_ROBOTICS_HARDWARE/lib:$PWD/_install/lib:$LD_LIBRARY_PATH"
ldd _install/bin/src_executor | grep 'not found' && exit 1 || true

systemctl status robotics.service --no-pager
sha256sum /apps/engineai_robotics/_install/bin/src_executor
```

## Controlled Startup

Executor startup is intentionally separate from motion triggering. Start the
custom executor only after the physical preflight is complete.

```bash
cd /home/user/projects/engineai_robotics_qualifier_20260905_per_motion
mkdir -p runtime_logs

sudo systemctl stop robotics.service
test "$(systemctl is-active robotics.service)" = inactive
test -z "$(pgrep -f '[s]rc_executor' || true)"

log="runtime_logs/custom_$(date +%Y%m%d_%H%M%S).log"
sudo setsid ./run_custom_robot_root.sh >"$log" 2>&1 < /dev/null &
sleep 5

pid=$(pgrep -n -f '^.*/_install/bin/src_executor t800$')
echo "$pid" > runtime_logs/custom_controller.pid
kill -0 "$pid"
pgrep -af src_executor
tail -n 100 "$log"
```

The corrected restricted graph starts and remains at `idle`; it does not
automatically enter `passive`. Do not send a gamepad combination during initial observation.
Confirm stable sensor and motor communication, no configuration error, and no
repeated fault before any motion test.

## Motion Map

These are the only reachable bindings in the corrected integration candidate.
The complete compatibility table is in [T800 Control Mapping](T800_CONTROL_MAPPING.md).

| State | Entry state | Gamepad | Automatic return |
| --- | --- | --- | --- |
| `idle` | initial, or `passive` | `LB+START` | none |
| `passive` (damping) | `idle`, PD states, or active actions | `LB+RB` | none |
| `pd_stand` | `passive` | `LB+A` | none |
| `pd_stand_x` (prone preparation) | `passive` or `pd_stand` | `LB+X` | none |
| `pd_stand_y` (supine preparation) | `passive` or `pd_stand` | `LB+Y` | none |
| `qualifier_front_kick` | `pd_stand` | `RB+A` | `pd_stand` |
| `qualifier_straight_punch` | `pd_stand` | `RB+Y` | `pd_stand` |
| `qualifier_hook_punch` | `pd_stand` | `LB+B` | `pd_stand` |
| `qualifier_jab_left` | `pd_stand` | `RB+B` | `pd_stand` |
`pd_stand_x` and `pd_stand_y` are the official competition preparation poses.
EngineAI's accepted `rl_supine_to_stance` recovery returns to `walk` in its
qualified graph. Because this restricted graph does not yet carry that return
state, neither official recovery nor the older shared-policy custom recovery is
currently reachable.

The spinning kick did not pass the final acceptance gate and has no reachable
state or key. Test accepted motions individually, beginning with the least
energetic motion and the exact required initial pose.

## Keyboard Control From Windows

The custom input arbiter accepts `virtual_gamepad/gamepad_keys` over LCM in
robot mode. The deployment includes an ARM64 terminal publisher so the Windows
workstation can use its keyboard through the existing SSH connection without
installing LCM or PyQt locally:

```powershell
ssh -t user@192.168.0.163 `
  "cd /home/user/projects/engineai_robotics_qualifier_20260905_per_motion && ./tools/virtual_gamepad/t800_keyboard_control --arm"
```

Run without `--arm` first to verify that `task_state` is received. The tool
refuses input if the custom executor is absent, the state feed is stale, or the
requested transition is not allowed from the current state. Press `?` for its
map and `q` to exit. Its single-key map is:

| Key | Request |
| --- | --- |
| `p` | `passive` damping |
| `t` | `pd_stand` |
| `x` / `y` | official prone / supine PD preparation |
| `j` / `h` | left jab / hook punch |
| `c` / `f` | straight punch / front kick |
| `i` | `idle` |

The spinning kick has no keyboard shortcut because it failed the qualification
gate. This keyboard path is an override for controlled debugging, not a bypass
for motor readiness, emergency stop, or the physical exclusion zone.

## Monitoring

```bash
target=/home/user/projects/engineai_robotics_qualifier_20260905_per_motion
pid=$(cat "$target/runtime_logs/custom_controller.pid")

sudo kill -0 "$pid" && echo running
pgrep -af src_executor
tail -f "$target"/runtime_logs/custom_*.log
```

Stop immediately on non-finite-value faults, model or trajectory shape errors,
joint-limit warnings, initial-pose rejection, loss of IMU/motor traffic,
unexpected movement, repeated state transitions, or emergency-stop request.

## Stop and Roll Back

Stop the saved custom process group before restarting the vendor service:

```bash
target=/home/user/projects/engineai_robotics_qualifier_20260905_per_motion
pid=$(cat "$target/runtime_logs/custom_controller.pid")

sudo kill -- "-$pid"
for _ in $(seq 1 20); do
  sudo kill -0 "$pid" 2>/dev/null || break
  sleep 0.25
done
sudo kill -0 "$pid" 2>/dev/null && sudo kill -KILL -- "-$pid"

test -z "$(pgrep -f '[s]rc_executor' || true)"
sudo systemctl start robotics.service
systemctl is-active robotics.service
systemctl status robotics.service --no-pager -l
```

Do not use an unscoped `pkill` after restarting the vendor controller. Always
target the saved custom process group, verify that no executor remains, and only
then start `robotics.service`.

If custom startup exits before the PID check succeeds, inspect the log and
restore the vendor service immediately. Do not repeatedly restart into an
unexplained hardware or configuration fault.
