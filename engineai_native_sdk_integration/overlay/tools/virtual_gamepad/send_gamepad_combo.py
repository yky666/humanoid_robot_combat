#!/usr/bin/env python3
import argparse
import os
import sys
import time


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from lcm_msgs.data import GamepadKeys


DIGITAL_INDEX = {
    "LB": 0,
    "RB": 1,
    "A": 2,
    "B": 3,
    "X": 4,
    "Y": 5,
    "BACK": 6,
    "START": 7,
    "CROSS_X_UP": 8,
    "CROSS_X_DOWN": 9,
    "CROSS_Y_LEFT": 10,
    "CROSS_Y_RIGHT": 11,
}

COMBOS = {
    "pd_stand": ("LB", "A"),
    "walk": ("LB", "B"),
    "dance": ("RB", "B"),
    "front_kick": ("RB", "A"),
    "spinning_kick": ("RB", "X"),
    "straight_punch": ("RB", "Y"),
    "hook_punch": ("LB", "X"),
    "jab_left": ("LB", "Y"),
    "recovery_supine": ("BACK", "A"),
    "passive": ("LB", "RB"),
    "idle": ("LB", "START"),
}


def make_msg(keys):
    msg = GamepadKeys()
    msg.timestamp = int(time.time() * 1_000_000)
    for key in keys:
        msg.digital_states[DIGITAL_INDEX[key]] = 1
    return msg


def publish_combo(lcm_handle, channel, keys, duration, rate_hz, release_count):
    period = 1.0 / rate_hz
    deadline = time.monotonic() + duration

    while time.monotonic() < deadline:
        msg = make_msg(keys)
        lcm_handle.publish(channel, msg.encode())
        time.sleep(period)

    for _ in range(release_count):
        msg = make_msg(())
        lcm_handle.publish(channel, msg.encode())
        time.sleep(period)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send one virtual gamepad key combination to the EngineAI SDK over LCM."
    )
    parser.add_argument("combo", nargs="?", choices=sorted(COMBOS), help="Named combo to publish.")
    parser.add_argument("--list", action="store_true", help="List available named combos.")
    parser.add_argument("--duration", type=float, default=0.35, help="Seconds to hold the combo.")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Publish rate while holding the combo.")
    parser.add_argument("--release-count", type=int, default=5, help="Number of release messages to publish.")
    parser.add_argument(
        "--channel",
        default="virtual_gamepad/gamepad_keys",
        help="LCM channel consumed by input_command_arbiter_runner.",
    )
    parser.add_argument(
        "--lcm-url",
        default=os.environ.get("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=0"),
        help="LCM URL. Defaults to local multicast ttl=0, matching SDK sim configs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list:
        for name, keys in sorted(COMBOS.items()):
            print(f"{name:16s} {'+'.join(keys)}")
        return 0

    if not args.combo:
        print("error: combo is required unless --list is used", file=sys.stderr)
        return 2

    try:
        import lcm
    except ImportError as exc:
        print(
            "error: Python package 'lcm' is missing. Install python3-lcm or run scripts/setup_virtual_gamepad_venv.sh.",
            file=sys.stderr,
        )
        print(f"detail: {exc}", file=sys.stderr)
        return 1

    keys = COMBOS[args.combo]
    lcm_handle = lcm.LCM(args.lcm_url)
    publish_combo(lcm_handle, args.channel, keys, args.duration, args.rate_hz, args.release_count)
    print(f"sent {args.combo}: {'+'.join(keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
