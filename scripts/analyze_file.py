#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import librosa

from mellymell.pitch import detect_pitch, hz_to_note
from mellymell.segment import segment_notes, NoteSegment


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
    ap.add_argument("--segments", type=Path, default=None, help="Write note segments CSV here")
    ap.add_argument("--segments-json", type=Path, default=None, help="Write note segments JSON here")
    ap.add_argument("--plot-segments", action="store_true", help="Plot Melodyne-style note blobs")
    ap.add_argument("--min-seg-dur", type=float, default=0.05, help="Minimum segment duration (s)")
    ap.add_argument("--gap", type=float, default=0.03, help="Max gap between frames to stitch (s)")
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

    # Segment notes
    segs = segment_notes(
        np.asarray(times), np.asarray(freqs), np.asarray(confs), a4=args.tuning,
        min_seg_dur=args.min_seg_dur, gap=args.gap
    )

    if args.segments is not None:
        with open(args.segments, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["start_s", "end_s", "note", "median_cents", "mean_confidence"])  # header
            for s in segs:
                w.writerow([f"{s.start_s:.6f}", f"{s.end_s:.6f}", s.note, f"{s.median_cents:.1f}", f"{s.mean_confidence:.3f}"])
        print(f"Wrote segments CSV: {args.segments} ({len(segs)} segments)")

    if args.segments_json is not None:
        import json

        payload = [
            {
                "start_s": s.start_s,
                "end_s": s.end_s,
                "note": s.note,
                "median_cents": s.median_cents,
                "mean_confidence": s.mean_confidence,
            }
            for s in segs
        ]
        args.segments_json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote segments JSON: {args.segments_json}")

    if args.plot or args.plot_segments:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))
        if args.plot:
            plt.plot(times, freqs, label="f0 (Hz)", alpha=0.5)
        if args.plot_segments and segs:
            # Draw rectangles per segment at the note's center frequency and color by median cents
            from mellymell.pitch import note_to_hz
            for s in segs:
                # Parse note name and octave: note ends with 1 or 2 digits depending on octave
                name = s.note[:-1]
                octave = int(s.note[-1])
                y = note_to_hz(name, octave, a4=args.tuning)
                # Color by median cents deviation
                cents = abs(s.median_cents)
                if cents <= 10:
                    color = "#2ecc71"  # green
                elif cents <= 30:
                    color = "#f1c40f"  # yellow
                else:
                    color = "#e74c3c"  # red
                plt.hlines(y, s.start_s, s.end_s, colors=color, linewidth=6, alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Pitch and segments")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

