from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_midi(f: float, a4: float = 440.0) -> float:
    return 69.0 + 12.0 * math.log2(f / a4)


def midi_to_hz(m: float, a4: float = 440.0) -> float:
    return a4 * (2.0 ** ((m - 69.0) / 12.0))


def midi_to_note(m: float) -> Tuple[str, int]:
    m_rounded = int(round(m))
    note_index = m_rounded % 12
    octave = m_rounded // 12 - 1
    return NOTE_NAMES[note_index], octave


def hz_to_note(f: float, a4: float = 440.0) -> Tuple[str, int, float]:
    """Map frequency to (note_name, octave, cents).
    Returns cents deviation from equal-tempered pitch [-50, 50].
    """
    m = hz_to_midi(f, a4=a4)
    name, octave = midi_to_note(m)
    cents = (m - round(m)) * 100.0
    return name, octave, cents


def note_to_hz(name: str, octave: int, a4: float = 440.0) -> float:
    idx = NOTE_NAMES.index(name)
    midi = (octave + 1) * 12 + idx
    return midi_to_hz(midi, a4=a4)


@dataclass
class PitchResult:
    frequency: float
    confidence: float


def yin(frame: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 2000.0, threshold: float = 0.1) -> PitchResult:
    """Basic YIN implementation returning frequency and confidence.
    frame: mono float32/64 np array
    """
    x = frame.astype(np.float64)
    x = x - np.mean(x)
    N = len(x)

    # Difference function d(tau)
    max_tau = int(sr / fmin)
    min_tau = max(2, int(sr / fmax))
    max_tau = min(max_tau, N - 2)
    d = np.zeros(max_tau + 1)
    for tau in range(1, max_tau + 1):
        diff = x[: N - tau] - x[tau:]
        d[tau] = np.dot(diff, diff)

    # Cumulative mean normalized difference CMND
    cmnd = np.zeros_like(d)
    cmnd[0] = 1.0
    running_sum = 0.0
    for tau in range(1, max_tau + 1):
        running_sum += d[tau]
        cmnd[tau] = d[tau] * tau / (running_sum + 1e-12)

    # Absolute threshold
    tau = min_tau
    best_tau = 0
    for tau in range(min_tau, max_tau):
        if cmnd[tau] < threshold:
            # Parabolic interpolation around tau for better precision
            if 1 <= tau < max_tau:
                a = cmnd[tau - 1]
                b = cmnd[tau]
                c = cmnd[tau + 1]
                denom = 2 * (2 * b - a - c)
                if abs(denom) > 1e-12:
                    tau = tau + (c - a) / denom
            best_tau = tau
            break

    if best_tau == 0:
        # Fallback to minimum CMND
        tau = int(np.argmin(cmnd[min_tau:max_tau])) + min_tau
        best_tau = tau

    f0 = sr / float(best_tau)
    # Confidence from CMND (lower is better). Map to 0..1.
    conf = float(max(0.0, min(1.0, (0.5 - cmnd[int(round(best_tau))]) / 0.5)))
    return PitchResult(frequency=f0, confidence=conf)


def detect_pitch(
    frame: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    method: str = "yin",
) -> PitchResult:
    """Detect pitch on a single analysis frame.
    frame should be mono. If stereo, pass a mono mix beforehand.
    """
    if frame.ndim > 1:
        frame = np.mean(frame, axis=-1)
    # Hann window to reduce spectral leakage
    if len(frame) > 0:
        frame = frame * np.hanning(len(frame))
    if method == "yin":
        return yin(frame, sr, fmin=fmin, fmax=fmax)
    else:
        return yin(frame, sr, fmin=fmin, fmax=fmax)

