from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .pitch import hz_to_note


@dataclass
class NoteSegment:
    start_s: float
    end_s: float
    note: str
    median_cents: float
    mean_confidence: float


def segment_notes(
    times: np.ndarray,
    freqs: np.ndarray,
    confs: np.ndarray,
    a4: float = 440.0,
    min_seg_dur: float = 0.05,
    gap: float = 0.03,
    conf_threshold: float = 0.2,
) -> List[NoteSegment]:
    """Group framewise f0 into note segments.

    Rules:
    - Ignore frames with confidence < conf_threshold or non-positive frequency.
    - Start a new segment when the note name changes or a gap > gap occurs.
    - Summarize each segment with median cents and mean confidence.
    """
    times = np.asarray(times)
    freqs = np.asarray(freqs)
    confs = np.asarray(confs)

    segments: List[NoteSegment] = []
    cur_note: Optional[str] = None
    cur_start: Optional[float] = None
    cur_cents: List[float] = []
    cur_confs: List[float] = []
    last_time: Optional[float] = None

    def close_segment(end_time: float):
        nonlocal cur_note, cur_start, cur_cents, cur_confs
        if cur_note is None or cur_start is None:
            return
        dur = end_time - cur_start
        if dur >= min_seg_dur and len(cur_cents) > 0:
            segments.append(
                NoteSegment(
                    start_s=cur_start,
                    end_s=end_time,
                    note=cur_note,
                    median_cents=float(np.median(cur_cents)),
                    mean_confidence=float(np.mean(cur_confs) if cur_confs else 0.0),
                )
            )
        # reset
        cur_note = None
        cur_start = None
        cur_cents = []
        cur_confs = []

    for t, f, c in zip(times, freqs, confs):
        valid = (f > 0) and (c >= conf_threshold)
        if not valid:
            # If we were in a segment, check gap
            if cur_note is not None and last_time is not None and (t - last_time) > gap:
                close_segment(last_time)
            last_time = t
            continue

        name, octave, cents = hz_to_note(f, a4=a4)
        note_str = f"{name}{octave}"
        if cur_note is None:
            cur_note = note_str
            cur_start = t
            cur_cents = [cents]
            cur_confs = [c]
        else:
            if note_str != cur_note:
                # Note changed, close previous at last_time
                close_segment(last_time if last_time is not None else t)
                cur_note = note_str
                cur_start = t
                cur_cents = [cents]
                cur_confs = [c]
            else:
                cur_cents.append(cents)
                cur_confs.append(c)
        last_time = t

    # Close trailing segment
    if cur_note is not None and last_time is not None and cur_start is not None:
        close_segment(last_time)

    return segments

