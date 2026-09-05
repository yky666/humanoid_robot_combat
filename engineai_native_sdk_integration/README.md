# EngineAI Native SDK Integration

The official SDK remains an upstream dependency rather than a duplicated source
tree. This directory records all local code/config changes in two forms:

- a complete binary-capable Git patch;
- an overlay containing the resulting modified and newly added files.

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

Apply one variant to a clean checkout at its matching base revision:

```bash
git apply --binary /path/to/engineai_native_sdk_c8a04c9_working_tree.patch
```

The accepted official recovery policy is archived separately under
`results/t800_canonical_v1_20260902/recovery_official/`.

## Qualification Warning

The overlays intentionally preserve historical files named
`t800_qualifier_joint_policy.*` because this is a complete working-tree archive.
They predate the corrected physical gate and are not an accepted joint policy.
The canonical stand/action/stand joint training was blocked after spinning kick
failed; only entries in `results/.../qualification/best/` are approved for use.
