#!/usr/bin/env python3
"""Publish analog virtual-stick commands over LCM (same channel as the Qt gamepad)."""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from lcm_msgs.data import GamepadKeys  # noqa: E402

# analog_states: LT, RT, LX, LY, RX, RY  (matches VirtualGamepadWidget)
DIRS = {
    "stand": (0.0, 0.0, 0.0),
    "fwd": (0.40, 0.0, 0.0),
    "back": (-0.40, 0.0, 0.0),
    "left": (0.0, -0.40, 0.0),
    "right": (0.0, 0.40, 0.0),
    "yaw_ccw": (0.0, 0.0, -0.40),
    "yaw_cw": (0.0, 0.0, 0.40),
    "omni": (0.35, 0.25, 0.30),
}

# Named button combos copied from send_gamepad_combo.py
DIGITAL_INDEX = {
    "LB": 0,
    "RB": 1,
    "A": 2,
    "B": 3,
    "X": 4,
    "Y": 5,
    "BACK": 6,
    "START": 7,
}
COMBOS = {
    "pd_stand": ("LB", "A"),
    "walk": ("LB", "B"),
    "passive": ("LB", "RB"),
}


def make_stick(lx: float, ly: float, ry: float) -> GamepadKeys:
    msg = GamepadKeys()
    msg.timestamp = int(time.time() * 1_000_000)
    msg.analog_states[2] = float(lx)
    msg.analog_states[3] = float(ly)
    msg.analog_states[5] = float(ry)
    return msg


def make_combo(keys: tuple[str, ...]) -> GamepadKeys:
    msg = GamepadKeys()
    msg.timestamp = int(time.time() * 1_000_000)
    for key in keys:
        msg.digital_states[DIGITAL_INDEX[key]] = 1
    return msg


def publish(lcm_handle, channel: str, msg: GamepadKeys, duration: float, rate_hz: float) -> None:
    period = 1.0 / rate_hz
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        msg.timestamp = int(time.time() * 1_000_000)
        lcm_handle.publish(channel, msg.encode())
        time.sleep(period)


def main() -> int:
    parser = argparse.ArgumentParser(description="Send analog virtual-stick or mode combo over LCM.")
    parser.add_argument("name", nargs="?", help="Direction or combo: " + ", ".join(sorted(list(DIRS) + list(COMBOS))))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--lx", type=float)
    parser.add_argument("--ly", type=float)
    parser.add_argument("--ry", type=float, default=0.0, help="Right-stick Y (yaw)")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--channel", default="virtual_gamepad/gamepad_keys")
    parser.add_argument("--lcm-url", default=os.environ.get("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=0"))
    args = parser.parse_args()

    if args.list:
        print("directions:")
        for k, v in DIRS.items():
            print(f"  {k:10s} lx,ly,ry={v}")
        print("combos:")
        for k, v in COMBOS.items():
            print(f"  {k:10s} {'+'.join(v)}")
        return 0

    try:
        import lcm
    except ImportError as exc:
        print(f"error: python package lcm missing: {exc}", file=sys.stderr)
        return 1

    lcm_handle = lcm.LCM(args.lcm_url)
    if args.lx is not None or args.ly is not None:
        lx = 0.0 if args.lx is None else args.lx
        ly = 0.0 if args.ly is None else args.ly
        publish(lcm_handle, args.channel, make_stick(lx, ly, args.ry), args.duration, args.rate_hz)
        print(f"sent stick lx={lx} ly={ly} ry={args.ry} for {args.duration}s")
        return 0

    if not args.name:
        parser.error("name is required unless --list / --lx --ly")
    if args.name in COMBOS:
        publish(lcm_handle, args.channel, make_combo(COMBOS[args.name]), min(args.duration, 0.4), args.rate_hz)
        # release
        publish(lcm_handle, args.channel, GamepadKeys(), 0.1, args.rate_hz)
        print(f"sent combo {args.name}: {'+'.join(COMBOS[args.name])}")
        return 0
    if args.name in DIRS:
        lx, ly, ry = DIRS[args.name]
        publish(lcm_handle, args.channel, make_stick(lx, ly, ry), args.duration, args.rate_hz)
        publish(lcm_handle, args.channel, make_stick(0, 0, 0), 0.2, args.rate_hz)
        print(f"sent {args.name} lx={lx} ly={ly} ry={ry} for {args.duration}s")
        return 0
    parser.error(f"unknown name {args.name}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
