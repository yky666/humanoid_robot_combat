#!/usr/bin/env python3
"""Staged T800 fixed-guard watchdog.

Standing is already proven. This process:
  1. keeps dual-GPU training alive
  2. advances command stages when log gates pass
  3. on stall: play + ffmpeg frames + heuristic recovery
  4. after omni-walk: resume on <=10 cm rough terrain
  5. stops when terrain omni gait meets the final gate
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
TRAIN_TASK = "Tracking-Flat-T800-Fixed-Guard-72-v0"
TERRAIN_TASK = "Tracking-Rough-T800-Fixed-Guard-72-v0"
PLAY_TASK = "Tracking-Flat-T800-Fixed-Guard-72-Play-v0"
TERRAIN_PLAY_TASK = "Tracking-Rough-T800-Fixed-Guard-72-Play-v0"
TMUX_TRAIN = "t800_fg72"
TMUX_WATCH = "t800_fg72_watch"
POLL_SEC = 45
CONFIRM_ITERS = 40
STALL_SEC = {
    0: 20 * 60,
    1: 25 * 60,
    2: 25 * 60,
    3: 30 * 60,
    4: 35 * 60,
    5: 60 * 60,
}
MAX_STALLS = 4

STAGE_NAMES = (
    "standing_balance",
    "forward_warmup",
    "backward_expansion",
    "lateral_expansion",
    "yaw_mixed",
    "light_terrain",
)

# Play stick commands matching a virtual D-pad / joystick.
PLAY_CMD = {
    0: (0.0, 0.0, 0.0),
    1: (0.30, 0.0, 0.0),
    2: (-0.25, 0.0, 0.0),
    3: (0.0, 0.30, 0.0),
    4: (0.25, 0.20, 0.40),
    5: (0.30, 0.20, 0.30),
}

GATES = {
    0: {"timeout": 0.90, "ep_len": 800, "bad_ori": 0.08, "reward": 50.0},
    1: {"timeout": 0.70, "ep_len": 500, "bad_ori": 0.25, "reward": 20.0},
    2: {"timeout": 0.65, "ep_len": 450, "bad_ori": 0.30, "reward": 15.0},
    3: {"timeout": 0.65, "ep_len": 450, "bad_ori": 0.30, "reward": 15.0},
    4: {"timeout": 0.60, "ep_len": 400, "bad_ori": 0.30, "reward": 10.0},
    5: {"timeout": 0.55, "ep_len": 350, "bad_ori": 0.45, "reward": 5.0},
}

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ITER_RE = re.compile(
    r"Learning iteration\s+(\d+)/(\d+).*?"
    r"Mean reward:\s+([-\d.]+).*?"
    r"Mean episode length:\s+([-\d.]+).*?"
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
    return {
        "stage": 1,
        "command_scale": 1.0,
        "phase": "train",
        "task": TRAIN_TASK,
        "load_run": "2026-09-11_01-24-26_t800_fg72_baoquan_ground_20260911",
        "checkpoint": "model_1100.pt",
        "run_name": "",
        "log_path": "",
        "ckpt_dir": "",
        "stall_count": 0,
        "stage_entered_at": time.time(),
        "last_good_iter": 0,
        "status": "init",
        "history": [],
        "done": False,
        "block_reason": "",
    }


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def write_status(state: dict, extra: str = "") -> None:
    lines = [
        f"# FG72 pipeline ({now()})",
        "",
        f"- status: **{state.get('status')}**",
        f"- stage: **{state.get('stage')} {STAGE_NAMES[int(state.get('stage', 0))]}**",
        f"- command_scale: {state.get('command_scale')}",
        f"- task: `{state.get('task')}`",
        f"- run: `{state.get('run_name')}`",
        f"- checkpoint: `{state.get('checkpoint')}`",
        f"- stall_count: {state.get('stall_count')}",
        f"- done: {state.get('done')}",
        extra,
        "",
    ]
    STATUS_PATH.write_text("\n".join(lines), encoding="utf-8")


def sh(cmd: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, shell=True, text=True, capture_output=True, check=check
    )


def train_running() -> bool:
    out = sh("pgrep -f 'scripts/rsl_rl/train.py --task Tracking' || true").stdout.strip()
    return bool(out)


def latest_ckpt(ckpt_dir: Path) -> str | None:
    if not ckpt_dir.is_dir():
        return None
    pts = sorted(ckpt_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    return pts[-1].name if pts else None


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
                "timeout": float(m.group(5)),
                "bad_ori": float(m.group(6)),
            }
        )
    return rows


def last_n_pass(rows: list[dict], stage: int, n: int) -> bool:
    gate = GATES[stage]
    if len(rows) < max(8, n // 2):
        return False
    window = rows[-n:] if len(rows) >= n else rows[-max(8, len(rows) // 2) :]
    return all(
        r["timeout"] >= gate["timeout"]
        and r["ep_len"] >= gate["ep_len"]
        and r["bad_ori"] <= gate["bad_ori"]
        and r["reward"] >= gate["reward"]
        for r in window
    )


def collapsed(rows: list[dict]) -> bool:
    if len(rows) < 15:
        return False
    tail = rows[-12:]
    return (
        sum(r["ep_len"] < 120 or r["bad_ori"] > 0.8 or r["timeout"] < 0.05 for r in tail)
        >= 8
    )


def kill_train() -> None:
    sh(f"tmux kill-session -t {TMUX_TRAIN} 2>/dev/null || true")
    sh("pkill -f 'scripts/rsl_rl/train.py --task Tracking-' 2>/dev/null || true")
    time.sleep(8)


def start_train(state: dict) -> None:
    PIPE.mkdir(parents=True, exist_ok=True)
    run_name = state["run_name"]
    log_path = PIPE / f"{run_name}.log"
    ckpt_arg = ""
    resume_arg = ""
    if state.get("checkpoint") and state.get("load_run"):
        resume_arg = "--resume True"
        ckpt_arg = f"--load_run {state['load_run']} --checkpoint {state['checkpoint']}"
    script = PIPE / "train_current.sh"
    script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
source /home/sys01/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd {ROOT}
export PYTHONUNBUFFERED=1
export FG72_STATE_JSON={STATE_PATH}
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \\
  scripts/rsl_rl/train.py \\
  --task {state['task']} \\
  --num_envs 2048 \\
  --max_iterations 30000 \\
  --headless \\
  --distributed \\
  --device cuda \\
  --run_name {run_name} \\
  --logger tensorboard \\
  --log_project_name t800_boxing \\
  {resume_arg} {ckpt_arg} \\
  2>&1 | tee {log_path}
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    sh(f"tmux kill-session -t {TMUX_TRAIN} 2>/dev/null || true")
    sh(f"tmux new-session -d -s {TMUX_TRAIN} 'bash {script}'")
    state["log_path"] = str(log_path)
    state["status"] = "training"
    save_state(state)


def discover_ckpt_dir(run_name: str) -> Path | None:
    if not LOGS.is_dir():
        return None
    cands = sorted(LOGS.glob(f"*{run_name}"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def play_stage(state: dict, stage: int) -> Path | None:
    ckpt_dir = Path(state.get("ckpt_dir") or "")
    if not ckpt_dir.is_dir():
        found = discover_ckpt_dir(state.get("run_name", ""))
        if found:
            ckpt_dir = found
            state["ckpt_dir"] = str(ckpt_dir)
    ckpt = latest_ckpt(ckpt_dir) if ckpt_dir.is_dir() else None
    if ckpt is None:
        ckpt = state.get("checkpoint")
        ckpt_dir = LOGS / state.get("load_run", "")
    if ckpt is None or not (ckpt_dir / ckpt).is_file():
        return None
    vx, vy, wz = PLAY_CMD.get(stage, PLAY_CMD[4])
    out_dir = PIPE / "videos" / f"stage{stage}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    play_log = out_dir / "play.log"
    task = TERRAIN_PLAY_TASK if stage >= 5 or "Rough" in state.get("task", "") else PLAY_TASK
    env = os.environ.copy()
    env["FG_PLAY_STAGE"] = str(min(stage, 4) if stage < 5 else 4)
    env["FG_PLAY_VX"] = str(vx)
    env["FG_PLAY_VY"] = str(vy)
    env["FG_PLAY_WZ"] = str(wz)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        PYTHON,
        str(ROOT / "scripts/rsl_rl/play.py"),
        "--task",
        task,
        "--num_envs",
        "4",
        "--device",
        "cuda:0",
        "--load_run",
        ckpt_dir.name,
        "--checkpoint",
        ckpt,
        "--video",
        "--video_length",
        "400",
        "--headless",
        "--enable_cameras",
    ]
    with play_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True
        )
    videos = list((ckpt_dir / "videos" / "play").glob("*.mp4")) if (ckpt_dir / "videos").exists() else []
    # also search recently written mp4s
    videos += list(LOGS.glob("*/videos/play/*.mp4"))
    videos = sorted(set(videos), key=lambda p: p.stat().st_mtime)
    if not videos:
        return None
    mp4 = videos[-1]
    dest = out_dir / f"stage{stage}.mp4"
    shutil.copy2(mp4, dest)
    frames = out_dir / "frames"
    frames.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(dest), "-vf", "fps=2,scale=960:-1", str(frames / "f%03d.png")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    (out_dir / "diag.json").write_text(
        json.dumps(
            {
                "time": now(),
                "stage": stage,
                "checkpoint": ckpt,
                "play_rc": proc.returncode,
                "video": str(dest),
                "frames": str(frames),
                "n_frames": len(list(frames.glob("*.png"))),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    state["last_video"] = str(dest)
    state["last_frames"] = str(frames)
    save_state(state)
    return dest


def recover(state: dict, rows: list[dict]) -> None:
    stage = int(state["stage"])
    state["stall_count"] = int(state.get("stall_count", 0)) + 1
    state["history"].append(
        {
            "time": now(),
            "event": "stall",
            "stage": stage,
            "stall_count": state["stall_count"],
            "last": rows[-1] if rows else {},
        }
    )
    try:
        play_stage(state, stage)
    except Exception as exc:  # noqa: BLE001
        state["history"].append({"time": now(), "event": "play_failed", "error": str(exc)})
    if state["stall_count"] > MAX_STALLS:
        state["status"] = "blocked"
        state["block_reason"] = f"stage {stage} stalled {state['stall_count']} times"
        save_state(state)
        write_status(state, f"\nBlocked: {state['block_reason']}")
        return
    scale = float(state.get("command_scale", 1.0))
    ep = rows[-1]["ep_len"] if rows else 0.0
    if ep < 120:
        state["command_scale"] = max(0.40, scale * 0.70)
        if stage > 1:
            state["stage"] = stage - 1
    elif ep < 350:
        state["command_scale"] = max(0.50, scale * 0.85)
    else:
        state["command_scale"] = max(0.60, scale * 0.90)
    ckpt_dir = Path(state.get("ckpt_dir") or "")
    ckpt = latest_ckpt(ckpt_dir) if ckpt_dir.is_dir() else None
    if ckpt:
        state["load_run"] = ckpt_dir.name
        state["checkpoint"] = ckpt
    state["run_name"] = f"t800_fg72_auto_s{state['stage']}_r{state['stall_count']}_{time.strftime('%H%M%S')}"
    state["stage_entered_at"] = time.time()
    state["status"] = "recover_restart"
    save_state(state)
    kill_train()
    start_train(state)


def advance(state: dict, rows: list[dict]) -> None:
    stage = int(state["stage"])
    ckpt_dir = Path(state.get("ckpt_dir") or "") or discover_ckpt_dir(state.get("run_name", ""))
    if ckpt_dir:
        state["ckpt_dir"] = str(ckpt_dir)
        ckpt = latest_ckpt(ckpt_dir)
        if ckpt:
            state["checkpoint"] = ckpt
            state["load_run"] = Path(ckpt_dir).name
    try:
        play_stage(state, stage)
    except Exception as exc:  # noqa: BLE001
        state["history"].append({"time": now(), "event": "gate_play_failed", "error": str(exc)})
    nxt = stage + 1
    state["history"].append(
        {
            "time": now(),
            "event": "advance",
            "from": stage,
            "to": nxt,
            "last": rows[-1] if rows else {},
        }
    )
    state["stage"] = nxt
    state["stall_count"] = 0
    state["command_scale"] = min(1.0, float(state.get("command_scale", 1.0)) + 0.05)
    state["stage_entered_at"] = time.time()
    if nxt >= 5 and "Rough" not in state.get("task", ""):
        state["task"] = TERRAIN_TASK
        state["run_name"] = f"t800_fg72_auto_terrain_{time.strftime('%Y%m%d_%H%M%S')}"
        state["status"] = "terrain_restart"
        save_state(state)
        kill_train()
        start_train(state)
        return
    if nxt > 5:
        state["done"] = True
        state["status"] = "success"
        save_state(state)
        try:
            for s in (1, 2, 3, 4, 5):
                play_stage(state, s)
        except Exception:
            pass
        write_status(state, "\nOmnidirectional baoquan gait met the final gate.")
        return
    state["status"] = "advanced"
    save_state(state)


def tick(state: dict) -> dict:
    if state.get("done"):
        write_status(state, "\nPipeline complete.")
        return state
    if state.get("status") == "blocked":
        write_status(state, f"\n{state.get('block_reason')}")
        return state
    if not train_running():
        if not state.get("run_name"):
            state["run_name"] = f"t800_fg72_auto_s{state['stage']}_{time.strftime('%Y%m%d_%H%M%S')}"
        state["status"] = "restart_missing_train"
        save_state(state)
        start_train(state)
        time.sleep(5)
        return state
    log_path = Path(state.get("log_path") or "")
    rows = parse_log(log_path) if log_path else []
    if not rows and state.get("run_name"):
        alt = PIPE / f"{state['run_name']}.log"
        rows = parse_log(alt)
        if rows:
            state["log_path"] = str(alt)
    ckpt_dir = discover_ckpt_dir(state.get("run_name", ""))
    if ckpt_dir:
        state["ckpt_dir"] = str(ckpt_dir)
        ckpt = latest_ckpt(ckpt_dir)
        if ckpt:
            state["checkpoint"] = ckpt
            # keep load_run pointing at the live run so later resume is correct
            if state.get("status") == "training":
                state["load_run"] = ckpt_dir.name
    last = rows[-1] if rows else {}
    extra = (
        f"\n- last_iter: {last.get('iter')}\n"
        f"- reward: {last.get('reward')}\n"
        f"- ep_len: {last.get('ep_len')}\n"
        f"- timeout: {last.get('timeout')}\n"
        f"- bad_ori: {last.get('bad_ori')}\n"
        f"- last_video: `{state.get('last_video', '')}`\n"
    )
    stage = int(state["stage"])
    if rows and last_n_pass(rows, stage, CONFIRM_ITERS):
        # require a minimum dwell so we do not skip walking after a standing policy
        dwell = time.time() - float(state.get("stage_entered_at", time.time()))
        min_dwell = 8 * 60 if stage == 0 else 12 * 60
        if stage == 0:
            min_dwell = 0  # standing already proven
        if dwell >= min_dwell:
            advance(state, rows)
            write_status(load_state(), extra)
            return load_state()
    if rows and collapsed(rows):
        elapsed = time.time() - float(state.get("stage_entered_at", time.time()))
        if elapsed > 6 * 60:
            recover(state, rows)
            write_status(load_state(), extra)
            return load_state()
    elapsed = time.time() - float(state.get("stage_entered_at", time.time()))
    if rows and elapsed > STALL_SEC.get(stage, 30 * 60):
        gate = GATES[stage]
        tail = rows[-15:] if len(rows) >= 15 else rows
        survival_ok = all(
            r["timeout"] >= gate["timeout"] and r["ep_len"] >= gate["ep_len"]
            for r in tail
        )
        # Do not cut command scale while the robot already survives the
        # episode-length / timeout gates; wait for the remaining fall-rate.
        if collapsed(rows) or not survival_ok:
            recover(state, rows)
            write_status(load_state(), extra)
            return load_state()
    state["status"] = "training"
    save_state(state)
    write_status(state, extra)
    return state


def main() -> None:
    PIPE.mkdir(parents=True, exist_ok=True)
    state = load_state()
    # Standing already passed on the ground run.
    if int(state.get("stage", 0)) < 1:
        state["stage"] = 1
    if not state.get("checkpoint"):
        state["checkpoint"] = "model_1100.pt"
        state["load_run"] = "2026-09-11_01-24-26_t800_fg72_baoquan_ground_20260911"
    if not state.get("run_name"):
        state["run_name"] = f"t800_fg72_auto_s{state['stage']}_{time.strftime('%Y%m%d_%H%M%S')}"
    if not state.get("stage_entered_at"):
        state["stage_entered_at"] = time.time()
    save_state(state)
    write_status(state, "\nWatchdog started.")
    live = sh("pgrep -af 'scripts/rsl_rl/train.py --task Tracking' || true").stdout
    owned = bool(state.get("run_name")) and state["run_name"] in live
    if train_running() and not owned:
        ground = LOGS / state["load_run"]
        ckpt = latest_ckpt(ground) or state["checkpoint"]
        state["checkpoint"] = ckpt
        state["load_run"] = ground.name if ground.is_dir() else state["load_run"]
        save_state(state)
        kill_train()
        start_train(state)
    elif not train_running():
        start_train(state)
    while True:
        try:
            state = tick(load_state())
            if state.get("done") or state.get("status") == "blocked":
                break
        except Exception as exc:  # noqa: BLE001
            err = PIPE / "watchdog_error.log"
            with err.open("a", encoding="utf-8") as fh:
                fh.write(f"{now()} {exc}\n")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
