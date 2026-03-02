#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import queue
import sys
import time
from typing import Deque, Optional

import numpy as np
import sounddevice as sd

from mellymell.pitch import detect_pitch, hz_to_note, note_to_hz
from mellymell.backends import available_realtime_methods


def parse_note_string(note_str: str) -> tuple[str, int]:
    """Parse a note string like 'A4' or 'C#3' into (name, octave)."""
    for i in range(len(note_str) - 1, 0, -1):
        if not note_str[i].lstrip('-').isdigit():
            return note_str[:i + 1], int(note_str[i + 1:])
    raise ValueError(f"Cannot parse note string: {note_str!r}")


def parse_args():
    ap = argparse.ArgumentParser(description="Realtime pitch display (mic)")
    ap.add_argument("--device", type=str, default=None, help="Input device name or index")
    ap.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    ap.add_argument("--samplerate", type=int, default=48000)
    ap.add_argument("--block", type=int, default=2048, help="Block size (samples)")
    ap.add_argument("--tuning", type=float, default=440.0, help="A4 tuning Hz")
    ap.add_argument("--fmin", type=float, default=50.0)
    ap.add_argument("--fmax", type=float, default=2000.0)
    ap.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    ap.add_argument("--median", type=int, default=5, help="Median window (frames)")
    ap.add_argument("--hysteresis", type=float, default=10.0, help="Note change hysteresis in cents")
    rt_methods = sorted(m for m, ok in available_realtime_methods().items() if ok)
    ap.add_argument("--method", type=str, default="yin", choices=rt_methods,
                   help=f"Pitch detection algorithm (installed: {', '.join(rt_methods)})")
    ap.add_argument("--plot", action="store_true", help="Show a live frequency plot (matplotlib)")
    return ap.parse_args()


def list_devices_and_exit():
    devs = sd.query_devices()
    print("Input devices:")
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            print(f"  {i}: {d['name']} (in={d['max_input_channels']}, out={d['max_output_channels']})")
    sys.exit(0)


def main():
    args = parse_args()
    if args.list_devices:
        list_devices_and_exit()

    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            # Non-fatal driver warnings
            print(status, file=sys.stderr)
        q.put(indata.copy().reshape(frames, -1))

    stream = sd.InputStream(
        device=args.device,
        channels=1,
        samplerate=args.samplerate,
        blocksize=args.block,
        callback=callback,
    )

    # Smoothing state
    freq_hist: Deque[float] = collections.deque(maxlen=max(1, args.median))
    shown_note: Optional[str] = None
    shown_freq: Optional[float] = None

    if args.plot:
        import matplotlib.pyplot as plt

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_xlabel("Frame")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Live f0")
        line, = ax.plot([], [], lw=1.5)
        ax.set_ylim(0, args.fmax * 1.2)
        ax.set_xlim(0, 200)
        plot_data: Deque[float] = collections.deque(maxlen=200)

    # Create PESTO stream processor if needed
    pesto_processor = None
    if args.method == "pesto":
        from mellymell.backends import PestoStreamProcessor
        pesto_processor = PestoStreamProcessor(sr=args.samplerate)

    print(f"Using {args.method.upper()} algorithm - Press Ctrl+C to stop")
    try:
        with stream:
            while True:
                block = q.get()
                block = block.squeeze(-1).astype(np.float32)
                if pesto_processor is not None:
                    res = pesto_processor.process_frame(block)
                else:
                    res = detect_pitch(block, args.samplerate, fmin=args.fmin, fmax=args.fmax, method=args.method)
                f = float(res.frequency)
                conf = float(res.confidence)

                # Gate low-confidence frames
                if not (np.isfinite(f) and f > 0 and conf >= args.conf):
                    # Print no pitch but keep previous shown value
                    sys.stdout.write("\r(no pitch)                                ")
                    sys.stdout.flush()
                    if args.plot:
                        plot_data.append(0.0)
                        line.set_data(range(len(plot_data)), list(plot_data))
                        ax.set_xlim(max(0, len(plot_data) - 200), len(plot_data))
                        plt.pause(0.001)
                    continue

                # Rolling median smoothing
                freq_hist.append(f)
                f_med = float(np.median(freq_hist)) if len(freq_hist) > 0 else f
                name, octave, cents = hz_to_note(f_med, a4=args.tuning)
                cur_note = f"{name}{octave}"

                # Hysteresis on note change: if changing note, require cents to exceed threshold
                if shown_note is not None and cur_note != shown_note:
                    # compute cents of f_med relative to the previous shown note center
                    try:
                        prev_name, prev_octave = parse_note_string(shown_note)
                    except ValueError:
                        prev_name, prev_octave = name, octave
                    f_prev = note_to_hz(prev_name, prev_octave, a4=args.tuning)
                    # cents delta between f_med and previous center
                    cents_delta = 1200.0 * np.log2(f_med / f_prev)
                    if abs(cents_delta) < args.hysteresis:
                        # Keep showing old note until we cross hysteresis
                        cur_note = shown_note

                shown_note = cur_note
                shown_freq = f_med

                sys.stdout.write(
                    f"\r{f_med:7.2f} Hz  {shown_note}  {cents:+6.1f} cents  conf={conf:4.2f}    "
                )
                sys.stdout.flush()

                if args.plot:
                    plot_data.append(f_med)
                    line.set_data(range(len(plot_data)), list(plot_data))
                    ax.set_xlim(max(0, len(plot_data) - 200), len(plot_data))
                    plt.pause(0.001)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

