# Cleanup And Consolidation Report

## Preserved Inputs

- `../URKL_work.zip` remains byte-for-byte preserved.
- The verified extraction remains under `../baselines/imported/urkl_work/`.
- GMR, GVHMR, and whole-body-control repositories were not moved or modified.
- Existing EngineAI SDK worktrees remain in place and are exposed through
  symlinks under `../repositories/`; the symlinks do not duplicate their data.

## Consolidated Project Material

- Reusable scripts: `../scripts/`
- Imported and converted work files: `../worktrees/`
- MuJoCo and IsaacLab reports/videos: `../results/`
- Isolated Python environments: `../environments/`
- External runtime extracts: `../runtime_dependencies/`
- Canonical asset links: `../assets/`

## Quarantine

Disposable smoke directories and a duplicate SDK audit clone were moved under
`quarantine/` instead of being deleted. This keeps the workspace navigable while
retaining rollback evidence. Nothing in quarantine is used by the active r10
training queue.

The active canonical training artifacts stay in `whole_body_tracking` because
that repository owns the queue and checkpoint layout; `../assets/` contains a
stable link to them for project-level discovery.
