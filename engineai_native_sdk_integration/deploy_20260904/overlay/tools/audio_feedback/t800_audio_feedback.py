#!/usr/bin/env python3
"""Play non-blocking operator feedback received from the T800 controller."""

import math
import os
import queue
import socket
import struct
import subprocess
import tempfile
import threading
import wave
from datetime import datetime
from pathlib import Path


BIND_HOST = os.environ.get("T800_AUDIO_BIND_HOST", "0.0.0.0")
BIND_PORT = int(os.environ.get("T800_AUDIO_BIND_PORT", "45800"))
ALLOWED_SOURCE = os.environ.get("T800_AUDIO_ALLOWED_SOURCE", "192.168.0.163")
AUDIO_DEVICE = os.environ.get("T800_AUDIO_DEVICE", "plughw:CARD=DAC,DEV=0")
SAMPLE_RATE = 48000

KEY_EVENTS = {
    129: "idle",
    3: "passive",
    5: "pd_stand",
    17: "pd_stand_x",
    33: "pd_stand_y",
    384: "supine_to_stance",
    18: "walk",
    6: "front_kick",
    34: "straight_punch",
    9: "hook_punch",
    10: "jab_left",
}

FREQUENCIES = {
    "idle": 392,
    "passive": 440,
    "pd_stand": 523,
    "pd_stand_x": 587,
    "pd_stand_y": 659,
    "supine_to_stance": 740,
    "walk": 698,
    "qualifier_front_kick": 784,
    "qualifier_straight_punch": 831,
    "qualifier_hook_punch": 880,
    "qualifier_jab_left": 988,
    "front_kick": 784,
    "straight_punch": 831,
    "hook_punch": 880,
    "jab_left": 988,
    "test": 660,
}


def append_tone(samples: list[int], frequency: float, duration: float, amplitude: float = 0.20) -> None:
    frame_count = int(SAMPLE_RATE * duration)
    fade_count = min(int(SAMPLE_RATE * 0.01), frame_count // 2)
    for index in range(frame_count):
        envelope = 1.0
        if index < fade_count:
            envelope = index / fade_count
        elif index >= frame_count - fade_count:
            envelope = (frame_count - index - 1) / fade_count
        value = amplitude * envelope * math.sin(2.0 * math.pi * frequency * index / SAMPLE_RATE)
        samples.append(int(max(-1.0, min(1.0, value)) * 32767))


def append_silence(samples: list[int], duration: float) -> None:
    samples.extend([0] * int(SAMPLE_RATE * duration))


def create_prompt(path: Path, frequency: int, accepted: bool) -> None:
    samples: list[int] = []
    append_tone(samples, frequency, 0.09)
    if accepted:
        append_silence(samples, 0.04)
        append_tone(samples, min(frequency * 1.25, 1200), 0.11)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))


class AudioWorker:
    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, bool]] = queue.Queue(maxsize=32)
        self._prompt_dir = Path(tempfile.mkdtemp(prefix="t800_audio_feedback_"))
        self._paths: dict[tuple[str, bool], Path] = {}
        for name, frequency in FREQUENCIES.items():
            for accepted in (False, True):
                path = self._prompt_dir / f"{name}_{'accepted' if accepted else 'request'}.wav"
                create_prompt(path, frequency, accepted)
                self._paths[(name, accepted)] = path
        threading.Thread(target=self._run, name="audio-worker", daemon=True).start()

    def submit(self, name: str, accepted: bool) -> None:
        if name not in FREQUENCIES:
            name = "test"
        try:
            self._queue.put_nowait((name, accepted))
        except queue.Full:
            print("audio queue full; dropping prompt", flush=True)

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            path = self._paths[event]
            try:
                subprocess.run(
                    ["/usr/bin/aplay", "-q", "-D", AUDIO_DEVICE, str(path)],
                    check=False,
                    timeout=3,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                print(f"audio playback failed: {error}", flush=True)


def parse_event(payload: str) -> tuple[str, bool] | None:
    kind, separator, value = payload.strip().partition(" ")
    if not separator:
        return None
    if kind == "KEY":
        try:
            return KEY_EVENTS.get(int(value), "test"), False
        except ValueError:
            return None
    if kind == "STATE":
        return value, True
    if kind == "TEST":
        return "test", True
    return None


def main() -> None:
    worker = AudioWorker()
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((BIND_HOST, BIND_PORT))
    print(
        f"listening on {BIND_HOST}:{BIND_PORT}, source={ALLOWED_SOURCE or 'any'}, device={AUDIO_DEVICE}",
        flush=True,
    )
    while True:
        payload, address = server.recvfrom(256)
        if ALLOWED_SOURCE and address[0] != ALLOWED_SOURCE:
            print(f"ignored packet from {address[0]}", flush=True)
            continue
        text = payload.decode("ascii", errors="replace")
        event = parse_event(text)
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        print(f"{timestamp} source={address[0]} payload={text.strip()} event={event}", flush=True)
        if event is not None:
            worker.submit(*event)


if __name__ == "__main__":
    main()
