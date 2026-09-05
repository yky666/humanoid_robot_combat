# T800 Control And State Mapping

This page separates three interfaces that use similar button names but do not
share one state machine: the EngineAI Native SDK, the legacy vendor controller,
and this repository's restricted qualifier graph. Always identify the running
controller before using a combination.

## State Names

| Operator concept | Native SDK state | Meaning in the custom controller |
| --- | --- | --- |
| IDLE | `idle` | Initial state; no active motion control. The corrected graph remains here until a command is sent. |
| Damping | `passive` | Passive damping torque. This is not a standing task and is not an alias for stance. |
| PD stand | `pd_stand` | Stable upright posture maintained by PD control. |
| Prone PD preparation | `pd_stand_x` | Official URKL prone recovery preparation pose. |
| Supine PD preparation | `pd_stand_y` | Official URKL supine recovery preparation pose. |

The vendor UI label `stance` belongs to the vendor/legacy controller workflow.
The open Native SDK describes its corresponding states as `idle`, `passive`,
and `pd_stand`; the custom controller does not invent a `stance` alias.

## Restricted Qualifier Gamepad

Use a Logitech F710 in XInput/Xbox mode. The bindings below are read directly
from `assets/config/t800/task_motion/qualifier_robot.yaml` and apply only while
the independent custom executor is running.

| F710 combination | Target state or action | Allowed from | Return |
| --- | --- | --- | --- |
| `LB+START` | `idle` | `passive` | stays in IDLE |
| `LB+RB` | `passive` damping | IDLE, PD states, active actions | stays in damping |
| `LB+A` | `pd_stand` | damping or recovery PD poses | stays in PD stand |
| `LB+X` | `pd_stand_x`, prone preparation | damping, PD stand, supine preparation | stays in pose |
| `LB+Y` | `pd_stand_y`, supine preparation | damping, PD stand, prone preparation | stays in pose |
| `RB+A` | front kick | PD stand | PD stand |
| `RB+Y` | straight punch | PD stand | PD stand |
| `LB+B` | hook punch | PD stand | PD stand |
| `RB+B` | left jab | PD stand | PD stand |
There is no spinning-kick binding because that policy failed its acceptance
gate. The earlier custom shared-policy recovery binding is also disabled; the
official recovery is archived but remains unreachable until its accepted
`walk` return state is integrated and tested without changing that contract.

## SSH Keyboard Mapping

Start the guarded ARM64 publisher from Windows:

```powershell
ssh -t user@192.168.0.163 `
  "cd /home/user/projects/engineai_robotics_qualifier_20260905_pdprep && ./tools/virtual_gamepad/t800_keyboard_control --arm"
```

| Keyboard | Publishes | Result |
| --- | --- | --- |
| `i` | `LB+START` | IDLE |
| `p` | `LB+RB` | passive damping |
| `t` | `LB+A` | PD stand |
| `x` / `y` | `LB+X` / `LB+Y` | prone / supine PD preparation |
| `f` | `RB+A` | front kick |
| `c` | `RB+Y` | straight punch |
| `h` | `LB+B` | hook punch |
| `j` | `RB+B` | left jab |
| `?` / `q` | local only | help / quit publisher |

Run the publisher without `--arm` first. That mode only receives and prints the
current `task_state`; it cannot send commands.

## Official Default And Legacy Differences

The upstream Native SDK T800 default graph uses `LB+START` for IDLE, `LB+RB`
for damping, `LB+A` for PD stand, `LB+B` for walk, `RB+B` for dance, and
`START+D-pad up/down` for stand-up/lie-down. The qualifier graph intentionally
reuses `LB+B` and `RB+B` for approved combat actions, so the default SDK table
must not be used while the qualifier executor is active.

The older `engineai_humanoid` deployment documents another mapping:
`LB+BACK` motor disable, `LB+START` motor enable, `LB+B` bent-leg stance,
`LB+A` straight-leg stand, and `LB+X` RL locomotion. Those commands apply to
that controller, not this Native SDK qualifier graph. In particular, pressing
`LB+B` while the qualifier graph is in PD stand requests a hook punch, not
legacy stance.

Authoritative references:

- [EngineAI Native SDK state switching](https://github.com/engineai-robotics/engineai_robotics_native_sdk/blob/main/README.md#14-state-switching)
- [EngineAI humanoid legacy joystick control](https://github.com/engineai-robotics/engineai_humanoid#1312-joystick-control)
- [T800 developer documentation](https://engineai.com.cn/open/docs/t800-developer/03?product=t800)

## Safe Startup Order

1. Confirm exactly one controller process and verify motor, IMU, RC02, and emergency-stop readiness.
2. Start the custom executor and confirm it remains in `idle`.
3. Enter damping with `LB+RB`, then PD stand with `LB+A` only when the robot is in the required pose.
4. Trigger one accepted action from PD stand and wait for its automatic PD-stand return.
5. Return to damping or IDLE deliberately; neither state is a substitute for physical power isolation.
