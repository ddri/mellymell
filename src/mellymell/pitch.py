from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_midi(f: float, a4: float = 440.0) -> float:
    if f <= 0:
        raise ValueError(f"Frequency must be positive, got {f}")
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


def yin(frame: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 2000.0, threshold: float = 0.05) -> PitchResult:
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

    # Threshold-based search for first good minimum
    best_tau = 0
    for tau in range(min_tau, max_tau):
        if cmnd[tau] < threshold:
            # Parabolic interpolation around tau for better precision
            if 1 <= tau < max_tau - 1:
                a = cmnd[tau - 1]
                b = cmnd[tau]
                c = cmnd[tau + 1]
                denom = 2 * (2 * b - a - c)
                if abs(denom) > 1e-12:
                    tau = tau + (c - a) / denom
            best_tau = tau
            break

    if best_tau == 0:
        # Fallback to global minimum if no threshold crossing found
        min_idx = int(np.argmin(cmnd[min_tau:max_tau])) + min_tau
        best_tau = min_idx
        
        # Apply parabolic interpolation
        if 1 <= min_idx < max_tau - 1:
            a = cmnd[min_idx - 1]
            b = cmnd[min_idx]
            c = cmnd[min_idx + 1]
            denom = 2 * (2 * b - a - c)
            if abs(denom) > 1e-12:
                best_tau = min_idx + (c - a) / denom

    if best_tau <= 0:
        return PitchResult(frequency=0.0, confidence=0.0)

    f0 = sr / float(best_tau)
    # Confidence from CMND (lower is better). Map to 0..1.
    cmnd_idx = min(int(round(best_tau)), len(cmnd) - 1)
    conf = float(max(0.0, min(1.0, (0.5 - cmnd[cmnd_idx]) / 0.5)))
    return PitchResult(frequency=f0, confidence=conf)


def mpm(frame: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 2000.0, 
        threshold: float = 0.7) -> PitchResult:
    """McLeod Pitch Method implementation.
    
    Based on "A Smarter Way to Find Pitch" by McLeod & Wyvill (2005).
    Uses normalized squared difference function and peak picking.
    
    Args:
        frame: mono float32/64 np array
        sr: sample rate
        fmin: minimum frequency to detect
        fmax: maximum frequency to detect  
        threshold: peak selection threshold (typically 0.8-0.95)
    
    Returns:
        PitchResult with frequency and clarity-based confidence
    """
    x = frame.astype(np.float64)
    x = x - np.mean(x)
    N = len(x)
    
    if N < 4:
        return PitchResult(frequency=0.0, confidence=0.0)
    
    # Calculate tau range based on frequency limits
    min_tau = max(2, int(sr / fmax))
    max_tau = min(N // 2, int(sr / fmin))
    
    if min_tau >= max_tau:
        return PitchResult(frequency=0.0, confidence=0.0)
    
    # Compute autocorrelation-like function r(tau)
    # r(tau) = sum(x[i] * x[i+tau]) for i in range(N-tau)
    r = np.zeros(max_tau + 1)
    for tau in range(max_tau + 1):
        if tau < N:
            r[tau] = np.dot(x[:N-tau], x[tau:])

    # Compute normalized squared difference function (NSDF)
    # NSDF(tau) = 2*r(tau) / (r[0] + r[tau])
    nsdf = np.zeros(max_tau + 1)
    for tau in range(1, max_tau + 1):
        denominator = r[0] + r[tau]
        if denominator > 1e-12:
            nsdf[tau] = 2 * r[tau] / denominator
        else:
            nsdf[tau] = 0.0
    
    nsdf[0] = 1.0  # Perfect correlation at tau=0
    
    # Find all local maxima in NSDF
    peaks = []
    for tau in range(min_tau, max_tau):
        if (tau > 0 and tau < len(nsdf) - 1 and 
            nsdf[tau] > nsdf[tau-1] and nsdf[tau] > nsdf[tau+1]):
            peaks.append((tau, nsdf[tau]))
    
    if not peaks:
        return PitchResult(frequency=0.0, confidence=0.0)
    
    # Find the maximum peak value for thresholding
    max_peak_value = max(peak[1] for peak in peaks)
    
    # Apply threshold: select first peak above threshold * max_peak
    selected_peak = None
    threshold_value = threshold * max_peak_value

    for tau, peak_value in peaks:
        if peak_value >= threshold_value:
            selected_peak = (tau, peak_value)
            break
    
    if selected_peak is None:
        # Fallback: use the highest peak if none meet threshold
        selected_peak = max(peaks, key=lambda p: p[1])
    
    tau_estimate, clarity = selected_peak
    
    # Parabolic interpolation for sub-sample precision
    if 1 <= tau_estimate < len(nsdf) - 1:
        a = nsdf[tau_estimate - 1]
        b = nsdf[tau_estimate]
        c = nsdf[tau_estimate + 1]
        
        # Parabolic interpolation formula
        denom = 2 * (2 * b - a - c)
        if abs(denom) > 1e-12:
            tau_estimate = tau_estimate + (c - a) / denom
            
            # Recalculate clarity at interpolated position
            clarity = b + 0.25 * (c - a) ** 2 / (2 * b - a - c)
    
    # Convert tau to frequency
    if tau_estimate > 0:
        frequency = sr / tau_estimate
    else:
        frequency = 0.0
    
    # Ensure frequency is within bounds
    if frequency < fmin or frequency > fmax:
        return PitchResult(frequency=0.0, confidence=0.0)
    
    # Clarity is already a good confidence measure (0-1 range)
    confidence = float(max(0.0, min(1.0, clarity)))
    
    return PitchResult(frequency=frequency, confidence=confidence)


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
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive, got {sr}")
    if fmin <= 0:
        raise ValueError(f"fmin must be positive, got {fmin}")
    if fmin >= fmax:
        raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
    if frame.ndim > 1:
        frame = np.mean(frame, axis=-1)
    if len(frame) == 0:
        return PitchResult(frequency=0.0, confidence=0.0)
    # Hann window to reduce spectral leakage
    frame = frame * np.hanning(len(frame))
    if method == "yin":
        return yin(frame, sr, fmin=fmin, fmax=fmax)
    elif method == "mpm":
        return mpm(frame, sr, fmin=fmin, fmax=fmax)
    else:
        raise ValueError(f"Unknown method {method!r}, expected 'yin' or 'mpm'")

