"""ML and bulk pitch detection backends.

This module provides:
- ``detect_pitch_bulk`` – run any method (yin, mpm, pyin, crepe, pesto) on a
  full audio buffer and get aligned time/frequency/confidence arrays.
- ``detect_pitch_polyphonic`` – Basic Pitch wrapper returning polyphonic note
  events.
- ``available_methods`` / ``available_realtime_methods`` – probe which backends
  are installed.
- ``PestoStreamProcessor`` – stateful PESTO wrapper for realtime streaming.

Optional ML dependencies are imported lazily.  If a backend is missing, a
clear ``ImportError`` is raised pointing to the appropriate pip extra.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .pitch import PitchResult, detect_pitch

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BulkPitchResult:
    """Framewise pitch estimates for a full audio buffer."""
    times: np.ndarray
    frequencies: np.ndarray
    confidences: np.ndarray


@dataclass
class NoteEvent:
    """A single note event from polyphonic detection."""
    start_s: float
    end_s: float
    midi_pitch: int
    amplitude: float
    # Derived convenience fields filled by factory helper
    frequency: float = 0.0
    note: str = ""


@dataclass
class PolyphonicResult:
    """Container for polyphonic note events."""
    note_events: List[NoteEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Method classification constants
# ---------------------------------------------------------------------------

FRAME_METHODS = {"yin", "mpm"}
BULK_METHODS = {"yin", "mpm", "pyin", "crepe", "pesto"}
POLYPHONIC_METHODS = {"basic_pitch"}
REALTIME_METHODS = {"yin", "mpm", "pesto"}

# ---------------------------------------------------------------------------
# Lazy importers
# ---------------------------------------------------------------------------

def _import_torchcrepe():
    try:
        import torch  # noqa: F401
        import torchcrepe  # noqa: F401
        return torchcrepe, torch
    except ImportError as exc:
        raise ImportError(
            "torchcrepe is required for CREPE pitch detection. "
            "Install it with: pip install mellymell[crepe]"
        ) from exc


def _import_pesto():
    try:
        import torch  # noqa: F401
        import pesto  # noqa: F401
        return pesto, torch
    except ImportError as exc:
        raise ImportError(
            "pesto-pitch is required for PESTO pitch detection. "
            "Install it with: pip install mellymell[pesto]"
        ) from exc


def _import_basic_pitch():
    try:
        from basic_pitch import inference as bp_inference  # noqa: F401
        return bp_inference
    except ImportError as exc:
        raise ImportError(
            "basic-pitch is required for polyphonic pitch detection. "
            "Install it with: pip install mellymell[polyphonic]"
        ) from exc


# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------

def available_methods() -> Dict[str, bool]:
    """Return a dict of method name -> is_installed for all bulk methods."""
    result: Dict[str, bool] = {}
    # Frame-based methods are always available
    for m in ("yin", "mpm"):
        result[m] = True
    # pyin uses librosa which is a core dependency
    result["pyin"] = True
    # CREPE
    try:
        _import_torchcrepe()
        result["crepe"] = True
    except ImportError:
        result["crepe"] = False
    # PESTO
    try:
        _import_pesto()
        result["pesto"] = True
    except ImportError:
        result["pesto"] = False
    # Basic Pitch (polyphonic)
    try:
        _import_basic_pitch()
        result["basic_pitch"] = True
    except ImportError:
        result["basic_pitch"] = False
    return result


def available_realtime_methods() -> Dict[str, bool]:
    """Return methods suitable for realtime (frame-at-a-time) use."""
    result: Dict[str, bool] = {}
    for m in ("yin", "mpm"):
        result[m] = True
    try:
        _import_pesto()
        result["pesto"] = True
    except ImportError:
        result["pesto"] = False
    return result


# ---------------------------------------------------------------------------
# Bulk pitch detection
# ---------------------------------------------------------------------------

def detect_pitch_bulk(
    audio: np.ndarray,
    sr: int,
    method: str = "yin",
    *,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    hop: int = 1024,
    frame_size: int = 2048,
    crepe_model: str = "tiny",
) -> BulkPitchResult:
    """Detect pitch across a full audio buffer.

    Parameters
    ----------
    audio : 1-D float array (mono)
    sr : sample rate
    method : one of ``yin``, ``mpm``, ``pyin``, ``crepe``, ``pesto``
    fmin, fmax : frequency range
    hop : hop size in samples (used by frame methods and pyin)
    frame_size : analysis frame size (used by frame methods)
    crepe_model : CREPE model capacity (``tiny``, ``small``, ``medium``,
        ``large``, ``full``)

    Returns
    -------
    BulkPitchResult with aligned times, frequencies, and confidences arrays.
    """
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive, got {sr}")
    if method not in BULK_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Available: {sorted(BULK_METHODS)}"
        )

    audio = np.asarray(audio, dtype=np.float64).ravel()

    if method in FRAME_METHODS:
        return _bulk_frame(audio, sr, method, fmin, fmax, hop, frame_size)
    elif method == "pyin":
        return _bulk_pyin(audio, sr, fmin, fmax, hop, frame_size)
    elif method == "crepe":
        return _bulk_crepe(audio, sr, fmin, fmax, hop, crepe_model)
    elif method == "pesto":
        return _bulk_pesto(audio, sr, hop)
    # Should not reach here due to validation above
    raise ValueError(f"Unhandled method {method!r}")  # pragma: no cover


# -- internal dispatch helpers ----------------------------------------------

def _bulk_frame(
    audio: np.ndarray, sr: int, method: str,
    fmin: float, fmax: float, hop: int, frame_size: int,
) -> BulkPitchResult:
    n = len(audio)
    times_list: list[float] = []
    freqs_list: list[float] = []
    confs_list: list[float] = []
    for start in range(0, n - frame_size, hop):
        buf = audio[start : start + frame_size]
        res = detect_pitch(
            buf.astype(np.float32), sr, fmin=fmin, fmax=fmax, method=method,
        )
        times_list.append(start / sr)
        freqs_list.append(float(res.frequency))
        confs_list.append(float(res.confidence))
    return BulkPitchResult(
        times=np.array(times_list),
        frequencies=np.array(freqs_list),
        confidences=np.array(confs_list),
    )


def _bulk_pyin(
    audio: np.ndarray, sr: int,
    fmin: float, fmax: float, hop: int, frame_size: int,
) -> BulkPitchResult:
    import librosa

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop,
        frame_length=frame_size,
    )
    # librosa.pyin returns NaN for unvoiced frames
    f0 = np.nan_to_num(f0, nan=0.0)
    confidences = np.asarray(voiced_probs, dtype=np.float64)
    confidences = np.nan_to_num(confidences, nan=0.0)
    n_frames = len(f0)
    times = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=hop,
    )
    return BulkPitchResult(
        times=times,
        frequencies=f0,
        confidences=confidences,
    )


def _bulk_crepe(
    audio: np.ndarray, sr: int,
    fmin: float, fmax: float, hop: int, model: str,
) -> BulkPitchResult:
    torchcrepe, torch = _import_torchcrepe()

    # torchcrepe expects (batch, samples) float32 tensor
    audio_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    hop_ms = hop / sr * 1000.0  # torchcrepe uses step_size in ms

    # Use torchcrepe.predict for full buffer processing
    time_t, frequency_t, confidence_t, _ = torchcrepe.predict(
        audio_t,
        sr,
        hop_length=hop,
        fmin=fmin,
        fmax=fmax,
        model=model,
        return_periodicity=True,
        batch_size=2048,
        device="cpu",
    )

    times = time_t.squeeze(0).numpy()
    frequencies = frequency_t.squeeze(0).numpy()
    confidences = confidence_t.squeeze(0).numpy()

    return BulkPitchResult(
        times=times,
        frequencies=frequencies,
        confidences=confidences,
    )


def _bulk_pesto(
    audio: np.ndarray, sr: int, hop: int,
) -> BulkPitchResult:
    pesto_mod, torch = _import_pesto()

    audio_t = torch.tensor(audio, dtype=torch.float32)
    timesteps, pitch, confidence, _ = pesto_mod.predict(
        audio_t, sr, step_size=hop / sr * 1000.0,
    )

    times = timesteps.numpy()
    frequencies = pitch.numpy()
    confidences = confidence.numpy()

    return BulkPitchResult(
        times=times,
        frequencies=frequencies,
        confidences=confidences,
    )


# ---------------------------------------------------------------------------
# Polyphonic detection
# ---------------------------------------------------------------------------

def detect_pitch_polyphonic(
    audio_path: str,
) -> PolyphonicResult:
    """Run Basic Pitch on an audio file and return polyphonic note events.

    Parameters
    ----------
    audio_path : path to an audio file readable by Basic Pitch / librosa.

    Returns
    -------
    PolyphonicResult containing a list of NoteEvent objects.
    """
    from .pitch import midi_to_hz, midi_to_note

    bp_inference = _import_basic_pitch()

    _, midi_data, _ = bp_inference.predict(str(audio_path))

    events: List[NoteEvent] = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            midi_pitch = note.pitch
            freq = midi_to_hz(float(midi_pitch))
            name, octave = midi_to_note(float(midi_pitch))
            events.append(NoteEvent(
                start_s=note.start,
                end_s=note.end,
                midi_pitch=midi_pitch,
                amplitude=note.velocity / 127.0,
                frequency=freq,
                note=f"{name}{octave}",
            ))

    # Sort by start time
    events.sort(key=lambda e: e.start_s)
    return PolyphonicResult(note_events=events)


# ---------------------------------------------------------------------------
# PESTO realtime streaming processor
# ---------------------------------------------------------------------------

class PestoStreamProcessor:
    """Stateful wrapper around PESTO for frame-at-a-time realtime use.

    Usage::

        processor = PestoStreamProcessor(sr=48000)
        # In audio callback:
        result = processor.process_frame(block)
    """

    def __init__(self, sr: int = 48000):
        self._pesto, self._torch = _import_pesto()
        self.sr = sr

    def process_frame(self, block: np.ndarray) -> PitchResult:
        """Process a single audio block and return a PitchResult."""
        audio_t = self._torch.tensor(
            block.astype(np.float32), dtype=self._torch.float32,
        )
        timesteps, pitch, confidence, _ = self._pesto.predict(
            audio_t, self.sr, step_size=len(block) / self.sr * 1000.0,
        )
        # Take last frame prediction
        freq = float(pitch[-1]) if len(pitch) > 0 else 0.0
        conf = float(confidence[-1]) if len(confidence) > 0 else 0.0
        if not np.isfinite(freq) or freq <= 0:
            return PitchResult(frequency=0.0, confidence=0.0)
        return PitchResult(frequency=freq, confidence=conf)
