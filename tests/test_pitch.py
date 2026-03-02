import math

import numpy as np
import pytest

from mellymell.pitch import (
    detect_pitch,
    hz_to_midi,
    hz_to_note,
    midi_to_hz,
    midi_to_note,
    note_to_hz,
    PitchResult,
)
from mellymell.segment import segment_notes, NoteSegment


def gen_tone(freq, sr=48000, dur=1.0, wave="sine"):
    t = np.arange(int(sr * dur)) / sr
    if wave == "sine":
        y = np.sin(2 * np.pi * freq * t)
    elif wave == "saw":
        # simple band-limited-ish saw using harmonics cutoff
        y = np.zeros_like(t)
        max_h = int(sr / (2 * freq))
        for k in range(1, max(2, max_h)):
            y += (1 / k) * np.sin(2 * np.pi * k * freq * t)
        y *= (2 / np.pi)
    elif wave == "triangle":
        y = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * freq * t))
    else:
        raise ValueError("unknown wave")
    return y.astype(np.float32)


# ── Existing pitch detection tests ──────────────────────────────────

@pytest.mark.parametrize("freq", [220.0, 440.0, 880.0])
@pytest.mark.parametrize("wave", ["sine", "triangle"])
@pytest.mark.parametrize("method", ["yin", "mpm"])
def test_detect_pitch_tones(freq, wave, method):
    sr = 48000
    y = gen_tone(freq, sr=sr, dur=0.1, wave=wave)
    res = detect_pitch(y, sr, method=method)
    assert res.confidence >= 0.2
    assert abs(res.frequency - freq) / freq < 0.02  # within 2%


def test_mpm_basic_functionality():
    """Test MPM algorithm basic functionality."""
    sr = 48000
    freq = 440.0
    y = gen_tone(freq, sr=sr, dur=0.1, wave="sine")

    from mellymell.pitch import mpm
    res = mpm(y, sr)

    assert res.frequency > 0
    assert res.confidence > 0
    assert abs(res.frequency - freq) / freq < 0.02


# ── Utility function tests ──────────────────────────────────────────

class TestHzToMidi:
    def test_a4(self):
        assert hz_to_midi(440.0) == pytest.approx(69.0)

    def test_a3(self):
        assert hz_to_midi(220.0) == pytest.approx(57.0)

    def test_c4(self):
        # C4 = MIDI 60 = 261.626 Hz
        assert hz_to_midi(261.6256) == pytest.approx(60.0, abs=0.01)

    def test_custom_a4(self):
        assert hz_to_midi(442.0, a4=442.0) == pytest.approx(69.0)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            hz_to_midi(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            hz_to_midi(-100.0)


class TestMidiToHz:
    def test_a4(self):
        assert midi_to_hz(69.0) == pytest.approx(440.0)

    def test_a3(self):
        assert midi_to_hz(57.0) == pytest.approx(220.0)

    def test_c4(self):
        assert midi_to_hz(60.0) == pytest.approx(261.6256, rel=1e-4)

    def test_custom_a4(self):
        assert midi_to_hz(69.0, a4=442.0) == pytest.approx(442.0)


class TestMidiToNote:
    def test_a4(self):
        assert midi_to_note(69.0) == ("A", 4)

    def test_c4(self):
        assert midi_to_note(60.0) == ("C", 4)

    def test_c_sharp_3(self):
        assert midi_to_note(49.0) == ("C#", 3)


class TestHzToNote:
    def test_a4(self):
        name, octave, cents = hz_to_note(440.0)
        assert name == "A"
        assert octave == 4
        assert cents == pytest.approx(0.0, abs=0.01)

    def test_slightly_sharp(self):
        # A few Hz above A4 should be sharp
        name, octave, cents = hz_to_note(445.0)
        assert name == "A"
        assert octave == 4
        assert cents > 0

    def test_slightly_flat(self):
        name, octave, cents = hz_to_note(435.0)
        assert name == "A"
        assert octave == 4
        assert cents < 0


class TestNoteToHz:
    def test_a4(self):
        assert note_to_hz("A", 4) == pytest.approx(440.0)

    def test_c4(self):
        assert note_to_hz("C", 4) == pytest.approx(261.6256, rel=1e-4)

    def test_round_trip(self):
        """hz_to_note → note_to_hz should round-trip close to original."""
        original = 440.0
        name, octave, cents = hz_to_note(original)
        recovered = note_to_hz(name, octave)
        assert recovered == pytest.approx(original, rel=0.01)


class TestRoundTrips:
    def test_midi_round_trip(self):
        """hz_to_midi → midi_to_hz should round-trip exactly."""
        for freq in [220.0, 440.0, 880.0, 329.63]:
            midi = hz_to_midi(freq)
            recovered = midi_to_hz(midi)
            assert recovered == pytest.approx(freq, rel=1e-10)

    def test_midi_note_round_trip(self):
        """midi_to_note → note_to_hz → hz_to_midi should be consistent."""
        for midi_val in [60, 69, 72, 48]:
            name, octave = midi_to_note(float(midi_val))
            hz = note_to_hz(name, octave)
            recovered_midi = hz_to_midi(hz)
            assert recovered_midi == pytest.approx(float(midi_val), abs=0.01)


# ── Input validation tests ──────────────────────────────────────────

class TestInputValidation:
    def test_detect_pitch_sr_zero(self):
        frame = np.zeros(2048, dtype=np.float32)
        with pytest.raises(ValueError):
            detect_pitch(frame, sr=0)

    def test_detect_pitch_sr_negative(self):
        frame = np.zeros(2048, dtype=np.float32)
        with pytest.raises(ValueError):
            detect_pitch(frame, sr=-1)

    def test_detect_pitch_unknown_method(self):
        frame = np.zeros(2048, dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown method"):
            detect_pitch(frame, sr=48000, method="unknown")

    def test_detect_pitch_fmin_zero(self):
        frame = np.zeros(2048, dtype=np.float32)
        with pytest.raises(ValueError):
            detect_pitch(frame, sr=48000, fmin=0)

    def test_detect_pitch_fmin_gte_fmax(self):
        frame = np.zeros(2048, dtype=np.float32)
        with pytest.raises(ValueError):
            detect_pitch(frame, sr=48000, fmin=2000, fmax=50)


# ── Edge case tests ─────────────────────────────────────────────────

class TestEdgeCases:
    def test_silence(self):
        """All-zeros frame should not crash. Note: YIN may report a
        spurious pitch on silence — callers should gate on RMS energy."""
        frame = np.zeros(2048, dtype=np.float32)
        res = detect_pitch(frame, sr=48000)
        assert isinstance(res, PitchResult)
        assert np.isfinite(res.frequency)
        assert np.isfinite(res.confidence)

    def test_empty_frame(self):
        """Empty frame should return zero frequency."""
        frame = np.array([], dtype=np.float32)
        res = detect_pitch(frame, sr=48000)
        assert res.frequency == 0.0
        assert res.confidence == 0.0

    def test_stereo_input(self):
        """Stereo frame should be handled (averaged to mono)."""
        sr = 48000
        mono = gen_tone(440.0, sr=sr, dur=0.1)
        stereo = np.column_stack([mono, mono])
        res = detect_pitch(stereo, sr)
        assert res.frequency > 0

    def test_pitch_result_dataclass(self):
        r = PitchResult(frequency=440.0, confidence=0.95)
        assert r.frequency == 440.0
        assert r.confidence == 0.95


# ── Segment tests ───────────────────────────────────────────────────

class TestSegmentNotes:
    def test_constant_pitch_one_segment(self):
        """Constant A4 across all frames should produce one segment."""
        n = 20
        times = np.linspace(0.0, 1.0, n)
        freqs = np.full(n, 440.0)
        confs = np.full(n, 0.9)
        segs = segment_notes(times, freqs, confs)
        assert len(segs) == 1
        assert segs[0].note == "A4"
        assert segs[0].median_cents == pytest.approx(0.0, abs=1.0)

    def test_note_change_two_segments(self):
        """Switching from A4 to E5 should produce two segments."""
        n = 20
        times = np.linspace(0.0, 1.0, n)
        freqs = np.concatenate([np.full(10, 440.0), np.full(10, 659.26)])
        confs = np.full(n, 0.9)
        segs = segment_notes(times, freqs, confs)
        assert len(segs) == 2
        assert segs[0].note == "A4"
        assert segs[1].note == "E5"

    def test_low_confidence_filtered(self):
        """Frames below confidence threshold should be ignored."""
        n = 20
        times = np.linspace(0.0, 1.0, n)
        freqs = np.full(n, 440.0)
        confs = np.full(n, 0.05)  # below default 0.2 threshold
        segs = segment_notes(times, freqs, confs)
        assert len(segs) == 0

    def test_gap_splitting(self):
        """A large time gap should split into two segments."""
        times = np.array([0.0, 0.01, 0.02, 0.03, 0.04,
                          0.5, 0.51, 0.52, 0.53, 0.54])
        freqs = np.full(10, 440.0)
        confs = np.full(10, 0.9)
        # gap=0.03 default, the jump from 0.04 to 0.5 is >> 0.03
        # But the low-conf frame triggers gap check — use explicit gap with a
        # zero-freq frame in between
        times_with_gap = np.array([0.0, 0.01, 0.02, 0.03, 0.04,
                                   0.10,  # gap frame (no pitch)
                                   0.5, 0.51, 0.52, 0.53, 0.54])
        freqs_with_gap = np.array([440.0, 440.0, 440.0, 440.0, 440.0,
                                   0.0,
                                   440.0, 440.0, 440.0, 440.0, 440.0])
        confs_with_gap = np.array([0.9, 0.9, 0.9, 0.9, 0.9,
                                   0.0,
                                   0.9, 0.9, 0.9, 0.9, 0.9])
        segs = segment_notes(times_with_gap, freqs_with_gap, confs_with_gap,
                             min_seg_dur=0.01, gap=0.03)
        assert len(segs) == 2

    def test_min_duration_filtering(self):
        """Segments shorter than min_seg_dur should be dropped."""
        # 3 frames spanning 0.02s — below default 0.05 min_seg_dur
        times = np.array([0.0, 0.01, 0.02])
        freqs = np.full(3, 440.0)
        confs = np.full(3, 0.9)
        segs = segment_notes(times, freqs, confs, min_seg_dur=0.05)
        assert len(segs) == 0

    def test_empty_input(self):
        """Empty arrays should produce no segments."""
        segs = segment_notes(np.array([]), np.array([]), np.array([]))
        assert len(segs) == 0

    def test_segment_dataclass(self):
        seg = NoteSegment(start_s=0.0, end_s=1.0, note="A4",
                          median_cents=2.5, mean_confidence=0.9)
        assert seg.start_s == 0.0
        assert seg.end_s == 1.0
        assert seg.note == "A4"
        assert seg.median_cents == 2.5
        assert seg.mean_confidence == 0.9

    def test_zero_frequency_ignored(self):
        """Frames with f=0 should be treated as silence."""
        times = np.linspace(0.0, 1.0, 20)
        freqs = np.zeros(20)
        confs = np.full(20, 0.9)
        segs = segment_notes(times, freqs, confs)
        assert len(segs) == 0
