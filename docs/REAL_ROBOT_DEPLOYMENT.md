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
| Custom package | `/home/user/projects/engineai_robotics_qualifier_20260905` |
| Vendor package | `/apps/engineai_robotics` |
| MNN runtime | 2.9.5 |
| Policy contract | `obs[1,140] -> actions[1,25]`, float32 |

The package was built on ARM64, checked against the controller's hardware
libraries, transferred over the robot LAN, and verified with a 2,441-file
SHA-256 manifest. The vendor package was not modified.

See the [2026-09-05 deployment log](DEPLOYMENT_LOG_20260905.md) for the first
controlled startup and rollback record.

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
git apply --binary \
  /path/to/humanoid_robot_combat/engineai_native_sdk_integration/deploy_20260904/engineai_native_sdk_335c60e_deploy_working_tree.patch
```

The equivalent file overlay is under
`engineai_native_sdk_integration/deploy_20260904/overlay/`. Use either the patch
or the overlay, not both.

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
engineai_robotics_qualifier_20260905/
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
cd /home/user/projects/engineai_robotics_qualifier_20260905
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
cd /home/user/projects/engineai_robotics_qualifier_20260905
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

The restricted state graph starts at `idle` and configures an automatic path to
`passive`. Do not send a gamepad combination during initial observation.
Confirm stable sensor and motor communication, no configuration error, and no
repeated fault before any motion test.

## Motion Map

These bindings are present in the integration candidate; presence does not mean
that every policy passed the current qualification gate.

| State | Entry state | Gamepad | Automatic return |
| --- | --- | --- | --- |
| `pd_stand` | `passive` | `LB+A` | none |
| `qualifier_front_kick` | `pd_stand` | `RB+A` | `pd_stand` |
| `qualifier_spinning_kick` | `pd_stand` | `RB+X` | `pd_stand` |
| `qualifier_straight_punch` | `pd_stand` | `RB+Y` | `pd_stand` |
| `qualifier_hook_punch` | `pd_stand` | `LB+X` | `pd_stand` |
| `qualifier_jab_left` | `pd_stand` | `LB+Y` | `pd_stand` |
| `qualifier_recovery_supine` | `passive` | `BACK+A` | `passive` |

The spinning kick did not pass the final acceptance gate and must not be treated
as an approved hardware action. Test accepted motions individually, beginning
with the least energetic motion and the exact required initial pose.

## Monitoring

```bash
target=/home/user/projects/engineai_robotics_qualifier_20260905
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
target=/home/user/projects/engineai_robotics_qualifier_20260905
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
