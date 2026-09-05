# Humanoid Robot Combat

Reproducible motion-retargeting, reinforcement-learning, evaluation, and
real-robot deployment workspace for combat-style motion on the EngineAI T800.
The repository joins four parts of the pipeline that otherwise live in separate
upstream projects:

```text
human motion (AMASS/BVH)
  -> GMR retargeting
  -> T800 tracking trajectory
  -> IsaacLab / Whole Body Tracking PPO
  -> formal rollout and visual evaluation
  -> ONNX actor extraction and MNN conversion
  -> EngineAI Native SDK integration
  -> isolated T800 real-robot deployment
```

This is a research and deployment archive, not a drop-in replacement for the
vendor controller. Real-robot operation requires the safety and rollback
procedure in [Real-Robot Deployment](docs/REAL_ROBOT_DEPLOYMENT.md).

## Current Status

The canonical qualification protocol uses 64 environments x 5 batches = 320
rollouts and requires a success rate of at least 0.95. TensorBoard tail metrics
and visual playback are supporting evidence, not acceptance evidence.

| Motion | Result | Formal rollouts |
| --- | --- | ---: |
| Hook punch | Passed | 320/320 |
| Front kick | Passed | 320/320 |
| Straight punch | Passed | 320/320 |
| Left jab | Passed | 316/320 |
| Supine recovery | Passed, official EngineAI MNN | 320/320 |
| Spinning kick | Failed tail gate | No formal rollout |
| Stand/action/stand joint policy | Blocked | Not started |

The latest spinning-kick run (`r9`) passed timeout, anchor, episode-length, and
joint-error checks, but its end-effector termination rate was 4.58% against a
3% maximum. It remains an archived experiment and is not in `best/`.

The independent ARM64 package staged on the T800 controller is an integration
candidate. Successful compilation, model conversion, or executor startup does
not promote a policy to accepted status.

## Repository Map

| Path | Purpose | Documentation |
| --- | --- | --- |
| `GMR/` | Human-to-T800 motion retargeting and T800 robot assets | [GMR README](GMR/README.md) |
| `whole_body_tracking/` | IsaacLab environments, PPO training, playback, and evaluation | [WBT README](whole_body_tracking/README.md) |
| `engineai_native_sdk_integration/` | EngineAI SDK patches, overlays, converted policy, and trajectories | [SDK integration](engineai_native_sdk_integration/README.md) |
| `isaaclab_integration/` | Pinned IsaacLab launcher and script delta | [IsaacLab integration](isaaclab_integration/README.md) |
| `results/t800_canonical_v1_20260902/` | Canonical references, checkpoints, reports, policies, and media | [Results README](results/t800_canonical_v1_20260902/README.md) |
| `manifests/` | Exact upstream repositories and revisions | [Repository manifest](manifests/repositories.json) |
| `docs/` | Archive notes and real-robot operations | [Archive notes](docs/ARCHIVE_20260905.md) |

## Quick Start

### 1. Clone the complete archive

Git LFS is required for policies, checkpoints, motions, videos, and compressed
experiment archives.

```bash
git lfs install
git clone https://github.com/yky666/humanoid_robot_combat.git
cd humanoid_robot_combat
git lfs pull
```

Check the pinned dependency revisions before reconstructing an environment:

```bash
cat manifests/repositories.json
```

### 2. Verify the canonical results

```bash
cd results/t800_canonical_v1_20260902
sha256sum -c SHA256SUMS
```

### 3. Choose a workflow

For retargeting, trajectory conversion, smoke training, formal training,
playback, and policy export, follow the
[training Quick Start](whole_body_tracking/QUICKSTART.md).

For Native SDK reconstruction, policy conversion, ARM64 compilation, and T800
deployment, follow the
[SDK integration guide](engineai_native_sdk_integration/README.md) and the
[real-robot runbook](docs/REAL_ROBOT_DEPLOYMENT.md).

## Development Environments

The recorded workspace used two Conda environments:

| Environment | Main use |
| --- | --- |
| `gmr` | SMPL-X/AMASS and BVH retargeting, GMR pickle and CSV generation |
| `env_isaaclab` | T800 trajectory conversion, simulation, training, evaluation, and video capture |

The deployment target is ARM64 Ubuntu 22.04 with ROS 2 Humble. The exact
upstream revisions are pinned in `manifests/repositories.json`; local absolute
paths shown in historical commands are examples from the archived workstation
and should be adapted to a new checkout.

## Core Workflows

### Motion preparation

```text
AMASS/SMPL-X or BVH
  -> GMR T800 pickle
  -> T800 joint CSV
  -> 50 Hz tracking NPZ
  -> schema and visual validation
```

The T800 tracking NPZ contract includes `fps`, `joint_pos`, `joint_vel`,
`body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`.

### Training and evaluation

Training uses the T800 tracking tasks in `whole_body_tracking` with smoke runs
before formal PPO runs. Candidate policies are evaluated with fixed batch
counts, explicit termination thresholds, reference tracking metrics, and visual
playback. Accepted artifacts and failed experiments are stored separately.

### Runtime conversion

The deployment integration extracts the actor from the training ONNX export and
converts it with MNN 2.9.5. The runtime contract is:

```text
obs:     float32 [1, 140]
actions: float32 [1, 25]
```

The Native SDK runner adds reference-frame alignment, startup interpolation,
initial-pose rejection, action clipping, joint-limit margins, finite-value
checks, safe-command retention, and restricted state transitions.

### Real-robot deployment

The deployed candidate is isolated from the vendor installation:

```text
Official: /apps/engineai_robotics
Custom:   /home/user/projects/engineai_robotics_qualifier_20260905
```

Only one `src_executor` may run. Never bypass the process guard in
`run_robot.sh`, and always preserve a tested path back to `robotics.service`.
See [Real-Robot Deployment](docs/REAL_ROBOT_DEPLOYMENT.md) for the complete
preflight, launch, monitoring, and rollback sequence.

## Reproducibility and Integrity

- Large artifacts are stored with Git LFS according to `.gitattributes`.
- `results/t800_canonical_v1_20260902/SHA256SUMS` covers the canonical results.
- The T800 runtime package carries its own `DEPLOYMENT_MANIFEST.sha256`.
- SDK and IsaacLab modifications are retained as both overlays and binary-capable patches.
- Upstream repositories and base commits are recorded in `manifests/repositories.json`.

Build trees, virtual environments, simulator caches, WandB caches, and repeated
every-100-step checkpoints are deliberately excluded. Final checkpoints from
reported rounds, formal reports, canonical logs, and retained review media are
included.

## Documentation Index

- [End-to-end training Quick Start](whole_body_tracking/QUICKSTART.md)
- [Whole Body Tracking project guide](whole_body_tracking/README.md)
- [EngineAI Native SDK integration](engineai_native_sdk_integration/README.md)
- [T800 real-robot deployment and rollback](docs/REAL_ROBOT_DEPLOYMENT.md)
- [2026-09-05 deployment log](docs/DEPLOYMENT_LOG_20260905.md)
- [Canonical results and qualification evidence](results/t800_canonical_v1_20260902/README.md)
- [Archive scope and restore notes](docs/ARCHIVE_20260905.md)
- [Pinned repository revisions](manifests/repositories.json)

## Safety and Scope

Combat motions can produce fast, high-energy limb movement. Simulation success
does not establish hardware safety. Use a load-rated stand or harness, keep the
robot's feet clear, maintain an exclusion zone, assign an operator to the tested
emergency stop, and begin with executor initialization only. Trigger each motion
separately only after reviewing its current qualification status and initial
pose requirements.

Third-party components retain their upstream licenses. Consult the license files
inside each imported project before redistribution.
