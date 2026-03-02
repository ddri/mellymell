#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path

import numpy as np
import librosa

from mellymell.pitch import hz_to_note, note_to_hz
from mellymell.segment import segment_notes, NoteSegment
from mellymell.backends import (
    available_methods,
    detect_pitch_bulk,
    detect_pitch_polyphonic,
    BULK_METHODS,
)


def parse_note_string(note_str: str) -> tuple[str, int]:
    """Parse a note string like 'A4' or 'C#3' into (name, octave)."""
    for i in range(len(note_str) - 1, 0, -1):
        if not note_str[i].lstrip('-').isdigit():
            return note_str[:i + 1], int(note_str[i + 1:])
    raise ValueError(f"Cannot parse note string: {note_str!r}")


def parse_args():
    methods = available_methods()
    installed_bulk = [m for m in sorted(BULK_METHODS) if methods.get(m, False)]

    ap = argparse.ArgumentParser(description="Offline file pitch analysis")
    ap.add_argument("audio", type=Path, help="Path to audio file")
    ap.add_argument("--output", type=Path, default=Path("pitch.csv"), help="CSV output path")
    ap.add_argument("--samplerate", type=int, default=0, help="Target sample rate (0 = file native)")
    ap.add_argument("--hop", type=int, default=1024, help="Hop size in samples")
    ap.add_argument("--frame", type=int, default=2048, help="Frame size in samples")
    ap.add_argument("--tuning", type=float, default=440.0, help="A4 tuning Hz")
    ap.add_argument("--fmin", type=float, default=50.0)
    ap.add_argument("--fmax", type=float, default=2000.0)
    ap.add_argument(
        "--method",
        choices=installed_bulk,
        default="yin",
        help=f"Pitch detection algorithm (installed: {', '.join(installed_bulk)})",
    )
    ap.add_argument("--polyphonic", action="store_true",
                    help="Use Basic Pitch polyphonic detection instead of monophonic")
    ap.add_argument("--crepe-model", default="tiny",
                    choices=["tiny", "small", "medium", "large", "full"],
                    help="CREPE model capacity (only used with --method crepe)")
    ap.add_argument("--plot", action="store_true", help="Plot pitch over time (requires matplotlib)")
    ap.add_argument("--segments", type=Path, default=None, help="Write note segments CSV here")
    ap.add_argument("--segments-json", type=Path, default=None, help="Write note segments JSON here")
    ap.add_argument("--plot-segments", action="store_true", help="Plot Melodyne-style note blobs")
    ap.add_argument("--html", type=Path, default=None, help="Write a simple HTML report (embeds PNG plot)")
    ap.add_argument("--png", type=Path, default=None, help="Optional PNG path for plot (defaults next to HTML)")
    ap.add_argument("--min-seg-dur", type=float, default=0.05, help="Minimum segment duration (s)")
    ap.add_argument("--gap", type=float, default=0.03, help="Max gap between frames to stitch (s)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.hop <= 0:
        print(f"Error: --hop must be positive, got {args.hop}", file=sys.stderr)
        sys.exit(1)
    if args.fmin <= 0 or args.fmin >= args.fmax:
        print(f"Error: --fmin must be positive and less than --fmax", file=sys.stderr)
        sys.exit(1)
    if not args.audio.exists():
        print(f"Error: file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # ── Polyphonic branch ──────────────────────────────────────────────
    if args.polyphonic:
        print(f"Running Basic Pitch polyphonic detection on {args.audio}...")
        poly_result = detect_pitch_polyphonic(str(args.audio))
        events = poly_result.note_events
        print(f"Detected {len(events)} note events")

        # Convert NoteEvents to NoteSegments for reuse of existing output code
        segs = [
            NoteSegment(
                start_s=e.start_s,
                end_s=e.end_s,
                note=e.note,
                median_cents=0.0,  # polyphonic doesn't provide cents
                mean_confidence=e.amplitude,
            )
            for e in events
        ]

        # Write a simplified framewise CSV (one row per event)
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["start_s", "end_s", "note", "midi_pitch", "amplitude"])
            for e in events:
                w.writerow([
                    f"{e.start_s:.6f}", f"{e.end_s:.6f}",
                    e.note, e.midi_pitch, f"{e.amplitude:.3f}",
                ])
        print(f"Wrote {args.output} ({len(events)} events)")

        # Fall through to shared segment output / plotting below
        # Build dummy framewise arrays for compatibility (not needed for segments)
        times = np.array([])
        freqs = np.array([])

    # ── Monophonic branch (bulk API) ───────────────────────────────────
    else:
        y, sr = librosa.load(str(args.audio), sr=(None if args.samplerate == 0 else args.samplerate), mono=True)

        print(f"Running {args.method.upper()} on {args.audio} ({len(y)/sr:.2f}s, sr={sr})...")
        bulk = detect_pitch_bulk(
            y, sr,
            method=args.method,
            fmin=args.fmin,
            fmax=args.fmax,
            hop=args.hop,
            frame_size=args.frame,
            crepe_model=args.crepe_model,
        )

        times = bulk.times
        freqs = bulk.frequencies
        confs = bulk.confidences

        # Compute note names / cents for CSV
        notes = []
        cents_list = []
        for f in freqs:
            if f > 0:
                name, octave, cents = hz_to_note(f, a4=args.tuning)
                notes.append(f"{name}{octave}")
                cents_list.append(cents)
            else:
                notes.append("")
                cents_list.append(0.0)

        # Write framewise CSV
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "frequency_hz", "note", "cents", "confidence"])
            for t, f0, note, cents, conf in zip(times, freqs, notes, cents_list, confs):
                w.writerow([f"{t:.6f}", f"{f0:.3f}", note, f"{cents:.1f}", f"{conf:.3f}"])

        print(f"Wrote {args.output} ({len(times)} frames)")

        # Segment notes
        segs = segment_notes(
            times, freqs, confs, a4=args.tuning,
            min_seg_dur=args.min_seg_dur, gap=args.gap,
        )

    # ── Shared output: segments CSV/JSON and plotting ──────────────────
    if args.segments is not None:
        with open(args.segments, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["start_s", "end_s", "note", "median_cents", "mean_confidence"])
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

    if args.plot or args.plot_segments or args.html is not None or args.png is not None:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 5))
        ax = plt.gca()
        if args.plot and len(times) > 0:
            ax.plot(times, freqs, label="f0 (Hz)", alpha=0.5)
        if (args.plot_segments or args.html is not None or args.png is not None) and segs:
            # Draw rectangles per segment at the note's center frequency and color by median cents
            for s in segs:
                name, octave = parse_note_string(s.note)
                y = note_to_hz(name, octave, a4=args.tuning)
                cents = abs(s.median_cents)
                if cents <= 10:
                    color = "#2ecc71"  # green
                elif cents <= 30:
                    color = "#f1c40f"  # yellow
                else:
                    color = "#e74c3c"  # red
                ax.hlines(y, s.start_s, s.end_s, colors=color, linewidth=6, alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Pitch and segments")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        png_path = None
        if args.png is not None:
            png_path = args.png
        elif args.html is not None:
            png_path = args.html.with_suffix(".png")

        if png_path is not None:
            fig.savefig(png_path, dpi=150)
            print(f"Wrote plot PNG: {png_path}")

        if args.html is not None:
            # Build minimal HTML embedding the PNG and linking to CSV/JSON
            rel_img = png_path.name if png_path is not None else ""
            frame_csv = args.output.name if hasattr(args.output, "name") else str(args.output)
            seg_csv = args.segments.name if args.segments is not None else None
            seg_json = args.segments_json.name if args.segments_json is not None else None
            esc_audio = html.escape(str(args.audio))
            esc_img = html.escape(str(rel_img)) if rel_img else ""
            esc_frame_csv = html.escape(str(frame_csv))
            esc_seg_csv = html.escape(str(seg_csv)) if seg_csv else None
            esc_seg_json = html.escape(str(seg_json)) if seg_json else None
            html_lines = [
                "<!DOCTYPE html>",
                "<meta charset='utf-8'>",
                "<title>mellymell analysis report</title>",
                "<style>body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;} .meta{color:#666} .links a{margin-right:12px}</style>",
                "<h1>mellymell analysis</h1>",
                f"<p class='meta'>Audio: {esc_audio} | A4={args.tuning} Hz | hop={args.hop} | frame={args.frame} | method={args.method}</p>",
            ]
            if esc_img:
                html_lines.append(f"<img alt='Pitch segments' src='{esc_img}' style='max-width:100%;height:auto;border:1px solid #ddd' />")
            links = [f"<a href='{esc_frame_csv}'>framewise CSV</a>"]
            if esc_seg_csv:
                links.append(f"<a href='{esc_seg_csv}'>segments CSV</a>")
            if esc_seg_json:
                links.append(f"<a href='{esc_seg_json}'>segments JSON</a>")
            html_lines.append(f"<p class='links'>{' | '.join(links)}</p>")
            args.html.write_text("\n".join(html_lines))
            print(f"Wrote HTML report: {args.html}")

        if args.plot and args.html is None and args.png is None:
            plt.show()


if __name__ == "__main__":
    main()
