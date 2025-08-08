#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import librosa

from mellymell.pitch import detect_pitch, hz_to_note


def parse_args():
    ap = argparse.ArgumentParser(description="Offline file pitch analysis")
    ap.add_argument("audio", type=Path, help="Path to audio file")
    ap.add_argument("--output", type=Path, default=Path("pitch.csv"), help="CSV output path")
    ap.add_argument("--samplerate", type=int, default=0, help="Target sample rate (0 = file native)")
    ap.add_argument("--hop", type=int, default=1024, help="Hop size in samples")
    ap.add_argument("--frame", type=int, default=2048, help="Frame size in samples")
    ap.add_argument("--tuning", type=float, default=440.0, help="A4 tuning Hz")
    ap.add_argument("--fmin", type=float, default=50.0)
    ap.add_argument("--fmax", type=float, default=2000.0)
    ap.add_argument("--plot", action="store_true", help="Plot pitch over time (requires matplotlib)")
    return ap.parse_args()


def main():
    args = parse_args()
    y, sr = librosa.load(str(args.audio), sr=(None if args.samplerate == 0 else args.samplerate), mono=True)
    n = len(y)
    hop = args.hop
    frame = args.frame

    times = []
    freqs = []
    notes = []
    cents_list = []
    confs = []

    for start in range(0, n - frame, hop):
        end = start + frame
        buf = y[start:end]
        res = detect_pitch(buf, sr, fmin=args.fmin, fmax=args.fmax)
        f = float(res.frequency)
        c = float(res.confidence)
        times.append(start / sr)
        freqs.append(f)
        confs.append(c)
        if f > 0:
            name, octave, cents = hz_to_note(f, a4=args.tuning)
            notes.append(f"{name}{octave}")
            cents_list.append(cents)
        else:
            notes.append("")
            cents_list.append(0.0)

    # Write CSV
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "frequency_hz", "note", "cents", "confidence"])  # header
        for t, f0, note, cents, conf in zip(times, freqs, notes, cents_list, confs):
            w.writerow([f"{t:.6f}", f"{f0:.3f}", note, f"{cents:.1f}", f"{conf:.3f}"])

    print(f"Wrote {args.output} ({len(times)} frames)")

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(times, freqs, label="f0 (Hz)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Pitch over time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

