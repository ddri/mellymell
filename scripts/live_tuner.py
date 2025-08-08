#!/usr/bin/env python3
from __future__ import annotations

import argparse
import queue
import sys
import time

import numpy as np
import sounddevice as sd

from mellymell.pitch import detect_pitch, hz_to_note


def parse_args():
    ap = argparse.ArgumentParser(description="Realtime pitch display (mic)")
    ap.add_argument("--device", type=str, default=None, help="Input device name or index")
    ap.add_argument("--samplerate", type=int, default=48000)
    ap.add_argument("--block", type=int, default=2048, help="Block size (samples)")
    ap.add_argument("--tuning", type=float, default=440.0, help="A4 tuning Hz")
    ap.add_argument("--fmin", type=float, default=50.0)
    ap.add_argument("--fmax", type=float, default=2000.0)
    return ap.parse_args()


def main():
    args = parse_args()
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

    print("Press Ctrl+C to stop")
    try:
        with stream:
            while True:
                block = q.get()
                block = block.squeeze(-1).astype(np.float32)
                res = detect_pitch(block, args.samplerate, fmin=args.fmin, fmax=args.fmax)
                f = res.frequency
                conf = res.confidence
                if np.isfinite(f) and f > 0:
                    name, octave, cents = hz_to_note(f, a4=args.tuning)
                    sys.stdout.write(
                        f"\r{f:7.2f} Hz  {name}{octave:02d}  {cents:+6.1f} cents  conf={conf:4.2f}    "
                    )
                else:
                    sys.stdout.write("\r(no pitch)                                ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

