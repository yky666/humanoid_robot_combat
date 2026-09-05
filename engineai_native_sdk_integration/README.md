# EngineAI Native SDK Integration

The official SDK remains an upstream dependency rather than a duplicated source
tree. This directory records local code/config changes in two forms:

- a Git patch for reviewing modifications to upstream-tracked files;
- an authoritative overlay containing both modified and newly added files.

## Development Tree

- Upstream: `https://github.com/engineai-robotics/engineai_robotics_native_sdk.git`
- Base commit: `c8a04c90889d6f4967bc1a43c0716796a86122aa`
- Patch: `engineai_native_sdk_c8a04c9_working_tree.patch`
- Overlay: `overlay/`

## Deployment Tree

- Upstream: `https://github.com/engineai-robotics/engineai_robotics_native_sdk.git`
- Base commit: `335c60e88772c26c7852d0abd6b3c7439037dd8f`
- Patch: `deploy_20260904/engineai_native_sdk_335c60e_deploy_working_tree.patch`
- Overlay: `deploy_20260904/overlay/`
- Runtime launcher: `deploy_20260904/runtime/run_custom_robot_root.sh`
- Official recovery-pose reference: `urkl_exams@0d759376cba552b480f267042d5d069ad5d96b50`

The complete T800 network, build, package, launch, monitoring, and rollback
procedure is documented in
[T800 Real-Robot Deployment](../docs/REAL_ROBOT_DEPLOYMENT.md).

Apply one variant to a clean checkout at its matching base revision:

```bash
git apply --binary /path/to/engineai_native_sdk_c8a04c9_working_tree.patch
```

For the real-robot deployment variant:

```bash
git clone https://github.com/engineai-robotics/engineai_robotics_native_sdk.git
cd engineai_robotics_native_sdk
git checkout 335c60e88772c26c7852d0abd6b3c7439037dd8f
cp -a \
  /path/to/humanoid_robot_combat/engineai_native_sdk_integration/deploy_20260904/overlay/. \
  ./

python tools/validate_qualifier_bundle.py
```

The patch alone intentionally omits files that do not exist at the pinned
upstream revision. Use the overlay for reconstruction; keep the patch for
source review and do not apply both representations to the same checkout.

The deployment overlay includes the actor extraction and bundle validation
tools, the 140-observation/25-action runner, restricted T800 state graph,
per-motion MNN runtime models, and six archived reference trajectories. Only
the four accepted standing-action actors are reachable. ARM64 build products
are not committed because they depend on the matching controller SDK libraries.

The deployment graph additionally carries EngineAI's official `pd_stand_x`
(prone) and `pd_stand_y` (supine) recovery preparation poses. The staged
recovery package exposes the accepted public `supine_to_stance` task from
`passive`, using `START+D-pad up`, and automatically returns to `walk`. See the
[control mapping](../docs/T800_CONTROL_MAPPING.md) for physical and keyboard
bindings and the legacy-controller compatibility boundary.

`tools/virtual_gamepad/t800_keyboard_control.cc` provides a guarded ARM64
terminal publisher for keyboard control through SSH. Build it on the ARM64 host
with `tools/virtual_gamepad/build_t800_keyboard_control.sh`; binaries remain
deployment artifacts and are not committed.

The accepted official recovery policy is archived separately under
`results/t800_canonical_v1_20260902/recovery_official/`.

## Qualification Warning

The overlays intentionally preserve historical files named
`t800_qualifier_joint_policy.*` because this is a complete working-tree archive.
They predate the corrected physical gate and are not an accepted joint policy.
The canonical stand/action/stand joint training was blocked after spinning kick
failed; only entries in `results/.../qualification/best/` are approved for use.
