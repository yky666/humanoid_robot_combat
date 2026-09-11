#!/usr/bin/env python3
"""Staged FG72 tracking watchdog.

Keeps terrain omni training alive. Advances command_scale only when
velocity tracking is good. When full-stick error_xy <= 0.15, re-renders
with a large virtual stick, 1 env, and a wide world camera.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/mnt/data/yangky/test/humanoid_robot_combat/whole_body_tracking")
PIPE = ROOT / "results/t800_fixed_guard72_20260910/pipeline"
LOGS = ROOT / "logs/rsl_rl/t800_fixed_guard_velocity_72d"
STATE_PATH = PIPE / "state.json"
STATUS_PATH = PIPE / "status.md"
PYTHON = "/home/sys01/miniconda3/envs/env_isaaclab/bin/python"
TASK = "Tracking-Bump-T800-Fixed-Guard-72-v0"
PLAY_TASK = "Tracking-Bump-T800-Fixed-Guard-72-Play-v0"
TMUX_TRAIN = "t800_fg72"
POLL_SEC = 45
CONFIRM = 30
MIN_DWELL = 12 * 60
STALL_SEC = 50 * 60

# Larger stick for the final demo so displacement is visible.
RENDER_CMDS = {
    "fwd": (0.70, 0.00, 0.00),
    "back": (-0.55, 0.00, 0.00),
    "left": (0.00, 0.55, 0.00),
    "right": (0.00, -0.55, 0.00),
    "yaw_ccw": (0.00, 0.00, 0.70),
    "yaw_cw": (0.00, 0.00, -0.70),
    "omni": (0.55, 0.45, 0.50),
}

TRACK_LEVELS = [
    {"scale": 0.55, "err_xy": 0.22, "err_yaw": 0.28, "timeout": 0.52, "ep_len": 380},
    {"scale": 0.70, "err_xy": 0.18, "err_yaw": 0.24, "timeout": 0.52, "ep_len": 380},
    {"scale": 0.85, "err_xy": 0.16, "err_yaw": 0.20, "timeout": 0.52, "ep_len": 360},
    {"scale": 1.00, "err_xy": 0.15, "err_yaw": 0.18, "timeout": 0.55, "ep_len": 360},
]

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ITER_RE = re.compile(
    r"Learning iteration\s+(\d+)/(\d+).*?"
    r"Mean reward:\s+([-\d.]+).*?"
    r"Mean episode length:\s+([-\d.]+).*?"
    r"Metrics/base_velocity/error_vel_xy:\s+([-\d.]+).*?"
    r"Metrics/base_velocity/error_vel_yaw:\s+([-\d.]+).*?"
    r"Episode_Termination/time_out:\s+([-\d.]+).*?"
    r"Episode_Termination/bad_orientation:\s+([-\d.]+)",
    re.S,
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def load_state() -> dict:
    PIPE.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def write_status(state: dict, extra: str = "") -> None:
    lvl = int(state.get("track_level", 0))
    gate = TRACK_LEVELS[min(lvl, len(TRACK_LEVELS) - 1)]
    STATUS_PATH.write_text(
        f"# FG72 track pipeline ({now()})\n\n"
        f"- status: **{state.get('status')}**\n"
        f"- track_level: **{lvl}/{len(TRACK_LEVELS)-1}** scale={state.get('command_scale')}\n"
        f"- gate: err_xy≤{gate['err_xy']} err_yaw≤{gate['err_yaw']}\n"
        f"- task: `{state.get('task')}`\n"
        f"- run: `{state.get('run_name')}`\n"
        f"- checkpoint: `{state.get('checkpoint')}`\n"
        f"- done: {state.get('done')}\n"
        f"{extra}\n",
        encoding="utf-8",
    )


def sh(cmd: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, check=check)


def train_running() -> bool:
    if sh("tmux has-session -t =t800_fg72 2>/dev/null").returncode == 0:
        return True
    return bool(sh("pgrep -f '[s]cripts/rsl_rl/train.py --task Tracking-' || true").stdout.strip())


def latest_ckpt(ckpt_dir: Path) -> str | None:
    if not ckpt_dir.is_dir():
        return None
    pts = sorted(ckpt_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    return pts[-1].name if pts else None


def discover_ckpt_dir(run_name: str) -> Path | None:
    if not LOGS.is_dir() or not run_name:
        return None
    cands = sorted(LOGS.glob(f"*{run_name}"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def parse_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    text = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    rows = []
    for m in ITER_RE.finditer(text):
        rows.append(
            {
                "iter": int(m.group(1)),
                "max_iter": int(m.group(2)),
                "reward": float(m.group(3)),
                "ep_len": float(m.group(4)),
                "err_xy": float(m.group(5)),
                "err_yaw": float(m.group(6)),
                "timeout": float(m.group(7)),
                "bad_ori": float(m.group(8)),
            }
        )
    return rows


def last_n_pass(rows: list[dict], level: int, n: int) -> bool:
    gate = TRACK_LEVELS[level]
    if len(rows) < max(8, n // 2):
        return False
    window = rows[-n:] if len(rows) >= n else rows[-max(8, len(rows) // 2) :]
    return all(
        r["err_xy"] <= gate["err_xy"]
        and r["err_yaw"] <= gate["err_yaw"]
        and r["timeout"] >= gate["timeout"]
        and r["ep_len"] >= gate["ep_len"]
        for r in window
    )


def collapsed(rows: list[dict]) -> bool:
    if len(rows) < 12:
        return False
    tail = rows[-12:]
    return sum(r["ep_len"] < 120 or r["bad_ori"] > 0.8 or r["timeout"] < 0.05 for r in tail) >= 8


def start_train(state: dict) -> None:
    run_name = state["run_name"]
    log_path = PIPE / f"{run_name}.log"
    resume = ""
    if state.get("checkpoint") and state.get("load_run"):
        resume = f"--resume True --load_run {state['load_run']} --checkpoint {state['checkpoint']}"
    script = PIPE / "train_current.sh"
    script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
source /home/sys01/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd {ROOT}
export PYTHONUNBUFFERED=1
export FG72_STATE_JSON={STATE_PATH}
export FG72_WALK_PRIOR=1
export FG72_WALK_PRIOR_MIX=0.88
export FG72_WALK_PRIOR_MIX_END=0.30
export FG72_WALK_PRIOR_ANNEAL_STEPS=200000
export FG72_WALK_PRIOR_ARM_KEEP=1.0
export FG72_WALK_PRIOR_IMIT=0.08
export FG72_CMD_MODE=back_yaw
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \\
  scripts/rsl_rl/train.py --task {TASK} --num_envs 2048 --max_iterations 25000 \\
  --headless --distributed --device cuda --run_name {run_name} \\
  --logger tensorboard --log_project_name t800_boxing {resume} \\
  2>&1 | tee {log_path}
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    sh(f"tmux kill-session -t ={TMUX_TRAIN} 2>/dev/null || true")
    sh(f"tmux new-session -d -s {TMUX_TRAIN} 'bash {script}'")
    state["log_path"] = str(log_path)
    state["status"] = "training"
    save_state(state)


def render_big_stick(state: dict) -> None:
    ckpt_dir = Path(state.get("ckpt_dir") or "")
    if not ckpt_dir.is_dir():
        found = discover_ckpt_dir(state.get("run_name", ""))
        if found:
            ckpt_dir = found
            state["ckpt_dir"] = str(ckpt_dir)
    ckpt = latest_ckpt(ckpt_dir) if ckpt_dir.is_dir() else state.get("checkpoint")
    if not ckpt or not (ckpt_dir / ckpt).is_file():
        state["history"].append({"time": now(), "event": "render_skip", "note": "no ckpt"})
        return
    out = PIPE / "videos" / "omni_dirs_bigstick"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["FG_PLAY_WIDE"] = "1"
    for name, (vx, vy, wz) in RENDER_CMDS.items():
        env["FG_PLAY_VX"] = str(vx)
        env["FG_PLAY_VY"] = str(vy)
        env["FG_PLAY_WZ"] = str(wz)
        log = out / f"{name}.log"
        cmd = [
            PYTHON,
            str(ROOT / "scripts/rsl_rl/play.py"),
            "--task",
            PLAY_TASK,
            "--num_envs",
            "1",
            "--device",
            "cuda:0",
            "--load_run",
            ckpt_dir.name,
            "--checkpoint",
            ckpt,
            "--video",
            "--video_length",
            "900",
            "--headless",
            "--enable_cameras",
        ]
        with log.open("w", encoding="utf-8") as fh:
            subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
        vids = sorted((ckpt_dir / "videos" / "play").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if vids:
            dest = out / f"{name}.mp4"
            shutil.copy2(vids[-1], dest)
            state["last_video"] = str(dest)
    state["history"].append({"time": now(), "event": "render_big_stick", "ckpt": ckpt})
    save_state(state)


def tick(state: dict) -> dict:
    if state.get("done"):
        write_status(state, "\nTracking gate met; big-stick videos written.")
        return state
    if not train_running():
        start_train(state)
        time.sleep(8)
        return load_state()
    log_path = Path(state.get("log_path") or "")
    rows = parse_log(log_path) if log_path else []
    if not rows and state.get("run_name"):
        rows = parse_log(PIPE / f"{state['run_name']}.log")
    ckpt_dir = discover_ckpt_dir(state.get("run_name", ""))
    if ckpt_dir:
        state["ckpt_dir"] = str(ckpt_dir)
        ckpt = latest_ckpt(ckpt_dir)
        if ckpt:
            state["checkpoint"] = ckpt
            if state.get("status") == "training":
                state["load_run"] = ckpt_dir.name
    last = rows[-1] if rows else {}
    extra = (
        f"\n- last_iter: {last.get('iter')}\n"
        f"- reward: {last.get('reward')}\n"
        f"- ep_len: {last.get('ep_len')}\n"
        f"- err_xy: {last.get('err_xy')}\n"
        f"- err_yaw: {last.get('err_yaw')}\n"
        f"- timeout: {last.get('timeout')}\n"
        f"- bad_ori: {last.get('bad_ori')}\n"
        f"- last_video: `{state.get('last_video', '')}`\n"
    )
    level = int(state.get("track_level", 0))
    level = max(0, min(level, len(TRACK_LEVELS) - 1))
    # keep JSON scale aligned so DDP ranks match
    want = TRACK_LEVELS[level]["scale"]
    if abs(float(state.get("command_scale", 1.0)) - want) > 1e-6:
        state["command_scale"] = want
        save_state(state)
    if rows and last_n_pass(rows, level, CONFIRM):
        dwell = time.time() - float(state.get("stage_entered_at", time.time()))
        if dwell >= MIN_DWELL:
            if level >= len(TRACK_LEVELS) - 1:
                state["done"] = True
                state["status"] = "success"
                state["history"].append({"time": now(), "event": "track_done", "last": last})
                save_state(state)
                try:
                    render_big_stick(state)
                except Exception as exc:  # noqa: BLE001
                    state["history"].append({"time": now(), "event": "render_failed", "error": str(exc)})
                    save_state(state)
                write_status(load_state(), extra)
                return load_state()
            state["track_level"] = level + 1
            state["command_scale"] = TRACK_LEVELS[level + 1]["scale"]
            state["stage_entered_at"] = time.time()
            state["history"].append(
                {"time": now(), "event": "track_advance", "from": level, "to": level + 1, "last": last}
            )
            save_state(state)
    elif rows and collapsed(rows):
        elapsed = time.time() - float(state.get("stage_entered_at", time.time()))
        if elapsed > 8 * 60 and level > 0:
            state["track_level"] = level - 1
            state["command_scale"] = TRACK_LEVELS[level - 1]["scale"]
            state["stage_entered_at"] = time.time()
            state["history"].append({"time": now(), "event": "track_backoff", "to": level - 1, "last": last})
            save_state(state)
    elapsed = time.time() - float(state.get("stage_entered_at", time.time()))
    if rows and elapsed > STALL_SEC and not last_n_pass(rows, level, 15):
        # tracking stall: do not cut a surviving policy; just keep training
        state["status"] = "training_stall_wait"
    else:
        state["status"] = "training"
    save_state(state)
    write_status(state, extra)
    return state


def main() -> None:
    PIPE.mkdir(parents=True, exist_ok=True)
    state = load_state()
    state["stage"] = 6
    state["task"] = TASK
    state["command_mode"] = "back_yaw"
    state["done"] = False
    state["status"] = "training"
    state.setdefault("track_level", 0)
    state["command_scale"] = TRACK_LEVELS[int(state["track_level"])]["scale"]
    state.setdefault("history", [])
    if not state.get("stage_entered_at"):
        state["stage_entered_at"] = time.time()
    save_state(state)
    write_status(state, "\nTrack watchdog started.")
    if not train_running():
        start_train(state)
    while True:
        try:
            state = tick(load_state())
            if state.get("done"):
                break
        except Exception as exc:  # noqa: BLE001
            with (PIPE / "watchdog_error.log").open("a", encoding="utf-8") as fh:
                fh.write(f"{now()} {exc}\n")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
