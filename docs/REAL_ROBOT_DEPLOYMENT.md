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
| Primary full candidate | `/home/user/projects/engineai_robotics_qualifier_20260905_recovery` (stopped) |
| Force-PD lab candidate | `/home/user/projects/engineai_robotics_qualifier_20260905_force_pd_lab` |
| Vendor package | `/apps/engineai_robotics` |
| MNN runtime | 2.9.5 |
| Policy contract | `obs[1,140] -> actions[1,25]`, float32 |
| Official PD reference | `urkl_exams@0d759376cba552b480f267042d5d069ad5d96b50` |
| IMU firmware | `V01.02.06b`, package `engineai-imu-update 1.0.3` |
| Last controller state | No `src_executor` running at 22:29 CST; vendor service inactive |

The package was built on ARM64, checked against the controller's hardware
libraries, transferred over the robot LAN, and verified with a 2,455-file
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

Verify the configured rule, actual listener socket, robot-LAN target, and
end-to-end SSH path separately. A rule printed by `netsh` does not prove that
Windows has bound its TCP listener.

### Windows Tailscale port forwarding

The recorded addresses and port mapping are:

| Endpoint | SSH account | Robot-LAN target | Windows Tailscale listener |
| --- | --- | --- | --- |
| Jetson Orin build host | `ubuntu` | `192.168.0.162:22` | `100.122.105.65:22162` |
| Nezha controller | `user` | `192.168.0.163:22` | `100.122.105.65:22163` |

Run the following commands in an **Administrator PowerShell** on the Windows
jump host. `100.74.87.113` is the recorded remote workstation's Tailscale IPv4
address; replace it if the authorized workstation's Tailscale address changes.

The repository includes an idempotent repair script that performs the complete
procedure below. It verifies the Tailscale listener address, removes stale
portproxy entries and duplicate firewall rules, recreates exactly one of each,
restarts IP Helper, and fails unless both TCP listener sockets appear:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\tools\windows\repair_t800_ssh_portproxy.ps1
```

To use different Tailscale addresses, pass them explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\tools\windows\repair_t800_ssh_portproxy.ps1 `
  -ListenAddress 100.122.105.65 `
  -AllowedRemoteAddress 100.74.87.113
```

First verify that Windows can reach both robot-side SSH servers and that the IP
Helper service required by `portproxy` is running:

```powershell
Test-NetConnection 192.168.0.162 -Port 22
Test-NetConnection 192.168.0.163 -Port 22

Get-Service iphlpsvc
Start-Service iphlpsvc
Set-Service iphlpsvc -StartupType Automatic
```

For a manual repair, remove the two old mappings first. This is safe for other
`portproxy` entries because it names only the two T800 listeners:

```powershell
netsh interface portproxy delete v4tov4 `
  listenaddress=100.122.105.65 listenport=22162
netsh interface portproxy delete v4tov4 `
  listenaddress=100.122.105.65 listenport=22163
```

Then create the two TCP forwarding rules:

```powershell
netsh interface portproxy add v4tov4 `
  listenaddress=100.122.105.65 listenport=22162 `
  connectaddress=192.168.0.162 connectport=22

netsh interface portproxy add v4tov4 `
  listenaddress=100.122.105.65 listenport=22163 `
  connectaddress=192.168.0.163 connectport=22
```

Delete any duplicate copies of the two named firewall rules, then recreate
exactly one rule for each listener. Restrict inbound access to the authorized
remote workstation rather than opening the ports to every Tailscale peer:

```powershell
Get-NetFirewallRule -DisplayName "T800 Orin SSH via Tailscale" `
  -ErrorAction SilentlyContinue | Remove-NetFirewallRule
Get-NetFirewallRule -DisplayName "T800 Nezha SSH via Tailscale" `
  -ErrorAction SilentlyContinue | Remove-NetFirewallRule

New-NetFirewallRule `
  -DisplayName "T800 Orin SSH via Tailscale" `
  -Direction Inbound -Action Allow -Protocol TCP `
  -LocalAddress 100.122.105.65 -LocalPort 22162 `
  -RemoteAddress 100.74.87.113 -Profile Any

New-NetFirewallRule `
  -DisplayName "T800 Nezha SSH via Tailscale" `
  -Direction Inbound -Action Allow -Protocol TCP `
  -LocalAddress 100.122.105.65 -LocalPort 22163 `
  -RemoteAddress 100.74.87.113 -Profile Any
```

Force IP Helper to reload the repaired mappings:

```powershell
Restart-Service iphlpsvc
Start-Sleep -Seconds 2
```

Check the configured mapping, actual listener sockets, and firewall rules:

```powershell
netsh interface portproxy show v4tov4

Get-NetTCPConnection -State Listen |
  Where-Object LocalPort -in 22162,22163 |
  Format-Table LocalAddress,LocalPort,OwningProcess

Get-NetFirewallRule -DisplayName "T800 * SSH via Tailscale" |
  Format-Table DisplayName,Enabled,Direction,Action

```

Seeing entries in `netsh interface portproxy show v4tov4` is not sufficient:
both ports must also appear in `Get-NetTCPConnection -State Listen`. Because the
firewall rules intentionally accept only source `100.74.87.113`, a
`Test-NetConnection` from the Windows jump host to its own Tailscale address is
not the end-to-end acceptance test and may be rejected by the source filter.

From the remote Linux workstation, connect through the forwarded ports:

```bash
# Verify the two TCP paths from the authorized Tailscale source first.
timeout 8 bash -lc '</dev/tcp/100.122.105.65/22162'
timeout 8 bash -lc '</dev/tcp/100.122.105.65/22163'

# Jetson Orin (.162)
ssh -p 22162 -o StrictHostKeyChecking=accept-new \
  ubuntu@100.122.105.65

# Nezha controller (.163)
ssh -p 22163 -o StrictHostKeyChecking=accept-new \
  user@100.122.105.65
```

Use the same port numbers with `scp` when transferring files:

```bash
scp -P 22162 ./artifact.tar.gz ubuntu@100.122.105.65:/home/ubuntu/
scp -P 22163 ./artifact.tar.gz user@100.122.105.65:/home/user/
```

If the rules are present but neither port is listening, confirm `iphlpsvc` is
running, confirm `100.122.105.65` is still assigned to the Tailscale adapter,
then recreate only these two rules and restart IP Helper. If the robot-LAN
target tests fail, repair the Windows Ethernet route or robot-side SSH service:

```powershell
Get-Service iphlpsvc
Get-NetIPAddress -AddressFamily IPv4 -IPAddress 100.122.105.65

Test-NetConnection 192.168.0.162 -Port 22
Test-NetConnection 192.168.0.163 -Port 22

Get-NetTCPConnection -State Listen |
  Where-Object LocalPort -in 22162,22163
```

To remove this exposure when it is no longer needed, run:

```powershell
netsh interface portproxy delete v4tov4 `
  listenaddress=100.122.105.65 listenport=22162
netsh interface portproxy delete v4tov4 `
  listenaddress=100.122.105.65 listenport=22163

Get-NetFirewallRule -DisplayName "T800 Orin SSH via Tailscale" |
  Remove-NetFirewallRule
Get-NetFirewallRule -DisplayName "T800 Nezha SSH via Tailscale" |
  Remove-NetFirewallRule
```

Do not place passwords, private keys, or `sshpass` commands in this repository.
The command syntax follows Microsoft's
[`netsh interface portproxy` reference](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/netsh-interface).

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
engineai_robotics_qualifier_20260905_recovery/
  _install/                         ARM64 binaries and runtime dependencies
  assets/config/t800/               base T800 and custom motion configuration
  assets/resource/                 T800 model XML, meshes, and environment
  DEPLOYMENT_MANIFEST.sha256       package integrity manifest
  DEPLOYMENT_NOTES.md              package identity and safety boundary
  run_robot.sh                     guarded vendor-compatible launcher
  run_custom_robot_root.sh         root launcher with isolated runtime paths
  tools/audio_feedback/            Orin USB-speaker listener and systemd unit
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
cd /home/user/projects/engineai_robotics_qualifier_20260905_recovery
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
cd /home/user/projects/engineai_robotics_qualifier_20260905_recovery
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

The full candidate retains official walking, both competition PD preparation
poses, the public supine-recovery definition, and all four accepted combat
policies. The force-PD lab is a separate package and is never a replacement for
this graph. The state paths are intentionally different:

```text
upright: idle -> passive -> pd_stand -> walk or combat -> pd_stand
supine:  pd_stand_y -> passive -> supine_to_stance -> walk -> pd_stand
prone:   pd_stand_x -> passive -> [public prone recovery unavailable]
```

Do not insert `pd_stand` before `supine_to_stance`: the former assumes an
upright base and two foot contacts, while the latter accepts entry from
`passive` with a supine base/contact state.

| State | Entry state | Gamepad | Automatic return |
| --- | --- | --- | --- |
| `idle` | initial, or `passive` | `LB+START` | none |
| `passive` (damping) | `idle`, PD states, or active actions | `LB+RB` | none |
| `pd_stand` | `passive` | `LB+A` | none |
| `pd_stand_x` (prone preparation) | `passive` or `pd_stand` | `LB+X` | none |
| `pd_stand_y` (supine preparation) | `passive` or `pd_stand` | `LB+Y` | none |
| `supine_to_stance` | quarantined; intended from `passive` | `START+D-pad up` | `walk` if requalified |
| `walk` (official SDK policy) | `pd_stand` | `RB+X` | none |
| `qualifier_front_kick` | `pd_stand` | `RB+A` | `pd_stand` |
| `qualifier_straight_punch` | `pd_stand` | `RB+Y` | `pd_stand` |
| `qualifier_hook_punch` | `pd_stand` | `LB+B` | `pd_stand` |
| `qualifier_jab_left` | `pd_stand` | `RB+B` | `pd_stand` |
`pd_stand_x` and `pd_stand_y` are the official competition preparation poses.
EngineAI's accepted `rl_supine_to_stance` definition remains staged in the
`_recovery` package and returns to `walk`, but its incoming FSM edge is
quarantined. The older shared-policy recovery and all prone recovery candidates
remain unreachable.

The state and assets are retained for reproducibility, but do not trigger
`supine_to_stance` on hardware until the observed greater-than-8-rad desired
targets have been explained and hard joint/output guards have passed renewed
simulation.

The spinning kick did not pass the final acceptance gate and has no reachable
state or key. Test accepted motions individually, beginning with the least
energetic motion and the exact required initial pose.

## Opt-In Force-PD Lab

The isolated force-PD configuration is for a harnessed joint-alignment test,
not dynamic floor recovery. It exposes only `idle`, `passive`, `pd_stand`,
`pd_stand_x`, and `pd_stand_y`; walking, combat, and `supine_to_stance` are
unreachable. Validate it with:

```bash
cd /home/user/projects/engineai_robotics_qualifier_20260905_force_pd_lab
python3 tools/validate_force_pd_lab.py
sha256sum -c DEPLOYMENT_MANIFEST.sha256
```

This package sets `strict_motion_check: false`, but a failed PD bias check is
overridden only while `LT` is held above 0.8. From `passive`, hold `LT`, then
hold `LB+A` to request forced `pd_stand`. A normal `LB+A` request retains the
bias rejection. The three-second quintic interpolation and configured PD gains
remain active; this does not add a contact-aware stand-up controller.

Before launch, verify that the independent copy actually selects the lab files:

```bash
grep -A4 -n 'tag: motion_task' assets/config/t800/mode.yaml
grep -A4 -n 'tag: global_options' assets/config/t800/mode.yaml
```

The expected scopes are `task_motion/force_pd_lab` and
`global_options/force_pd_lab`. A package selecting `qualifier_robot` is not the
isolated lab and must not be launched as one.

## Keyboard Control From Windows

The custom input arbiter accepts `virtual_gamepad/gamepad_keys` over LCM in
robot mode. The deployment includes an ARM64 terminal publisher so the Windows
workstation can use its keyboard through the existing SSH connection without
installing LCM or PyQt locally:

```powershell
ssh -t user@192.168.0.163 `
  "cd /home/user/projects/engineai_robotics_qualifier_20260905_recovery && ./tools/virtual_gamepad/t800_keyboard_control --arm"
```

Run without `--arm` first to verify that `task_state` is received. The tool
refuses input if the custom executor is absent, the state feed is stale, or the
requested transition is not allowed from the current state. Press `?` for its
map and `q` to exit. Its single-key map is:

| Key | Request |
| --- | --- |
| `p` | `passive` damping |
| `t` | `pd_stand` |
| `w` | official `walk` mode |
| `x` / `y` | official prone / supine PD preparation |
| `u` | official `supine_to_stance`, from `passive` only |
| `j` / `h` | left jab / hook punch |
| `c` / `f` | straight punch / front kick |
| `i` | `idle` |

The spinning kick has no keyboard shortcut because it failed the qualification
gate. This keyboard path is an override for controlled debugging, not a bypass
for motor readiness, emergency stop, or the physical exclusion zone.

## Orin Audio Feedback

Nezha has no ALSA playback device. The Orin at `192.168.0.162` exposes the
robot USB DAC, so the custom input arbiter sends non-blocking UDP feedback to
port `45800`. Install the listener on the Orin after synchronizing the source
tree:

```bash
cd /home/ubuntu/engineai_robotics_native_sdk_t800_deploy_20260904
sudo cp tools/audio_feedback/t800-audio-feedback.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now t800-audio-feedback.service
systemctl status t800-audio-feedback.service --no-pager
```

A recognized combination produces one tone. A confirmed task-state change
produces a two-tone response at the corresponding pitch. Thus a command that
is received but rejected by the state graph produces only the first tone.
Inspect exact event names with:

```bash
journalctl -u t800-audio-feedback.service -n 100 --no-pager
```

## Monitoring

```bash
target=/home/user/projects/engineai_robotics_qualifier_20260905_recovery
pid=$(cat "$target/runtime_logs/custom_controller.pid")

sudo kill -0 "$pid" && echo running
pgrep -af src_executor
tail -f "$target"/runtime_logs/custom_*.log
```

For a read-only one-shot summary, copy the packaged helper to Nezha and run:

```bash
./check_t800_status.sh
```

It reports the vendor service, exact custom executable/cwd, last entered FSM
state from that process's log, and EtherCAT slave state. When no executor is
running, the FSM state is correctly reported as unavailable rather than
`passive` or `idle`.

Stop immediately on non-finite-value faults, model or trajectory shape errors,
joint-limit warnings, initial-pose rejection, loss of IMU/motor traffic,
unexpected movement, repeated state transitions, or emergency-stop request.

## Stop and Roll Back

Stop the saved custom process group before restarting the vendor service:

```bash
target=/home/user/projects/engineai_robotics_qualifier_20260905_recovery
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
