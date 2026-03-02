"""Tests for mellymell.backends — bulk pitch detection and ML backends."""
from __future__ import annotations

import numpy as np
import pytest

from mellymell.backends import (
    available_methods,
    available_realtime_methods,
    detect_pitch_bulk,
    detect_pitch_polyphonic,
    BulkPitchResult,
    NoteEvent,
    PolyphonicResult,
    BULK_METHODS,
    FRAME_METHODS,
    REALTIME_METHODS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gen_tone(freq: float, sr: int = 48000, dur: float = 1.0) -> np.ndarray:
    """Generate a sine tone."""
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


# ---------------------------------------------------------------------------
# TestAvailableMethods
# ---------------------------------------------------------------------------

class TestAvailableMethods:
    def test_always_present(self):
        """yin, mpm, and pyin should always be available."""
        m = available_methods()
        assert m["yin"] is True
        assert m["mpm"] is True
        assert m["pyin"] is True

    def test_all_keys_present(self):
        """All known methods should appear in the dict."""
        m = available_methods()
        for key in ("yin", "mpm", "pyin", "crepe", "pesto", "basic_pitch"):
            assert key in m

    def test_values_are_bool(self):
        m = available_methods()
        for v in m.values():
            assert isinstance(v, bool)

    def test_realtime_methods(self):
        """yin and mpm should always be available for realtime."""
        rt = available_realtime_methods()
        assert rt["yin"] is True
        assert rt["mpm"] is True


# ---------------------------------------------------------------------------
# TestBulkWithFrameMethods
# ---------------------------------------------------------------------------

class TestBulkWithFrameMethods:
    @pytest.mark.parametrize("method", ["yin", "mpm"])
    def test_returns_bulk_result(self, method):
        """Frame methods through bulk API should return aligned arrays."""
        sr = 48000
        audio = gen_tone(440.0, sr=sr, dur=0.5)
        result = detect_pitch_bulk(audio, sr, method=method)
        assert isinstance(result, BulkPitchResult)
        assert len(result.times) == len(result.frequencies)
        assert len(result.times) == len(result.confidences)
        assert len(result.times) > 0

    @pytest.mark.parametrize("method", ["yin", "mpm"])
    def test_accuracy_440(self, method):
        """Frame methods should detect 440 Hz accurately via bulk API."""
        sr = 48000
        audio = gen_tone(440.0, sr=sr, dur=0.5)
        result = detect_pitch_bulk(audio, sr, method=method)
        # Filter frames with non-zero frequency
        voiced = result.frequencies[result.frequencies > 0]
        assert len(voiced) > 0
        median_freq = np.median(voiced)
        assert abs(median_freq - 440.0) / 440.0 < 0.02  # within 2%


# ---------------------------------------------------------------------------
# TestBulkPyin
# ---------------------------------------------------------------------------

class TestBulkPyin:
    def test_pyin_returns_result(self):
        """pYIN should return a valid BulkPitchResult."""
        sr = 48000
        audio = gen_tone(440.0, sr=sr, dur=1.0)
        result = detect_pitch_bulk(audio, sr, method="pyin")
        assert isinstance(result, BulkPitchResult)
        assert len(result.times) > 0
        assert len(result.times) == len(result.frequencies)

    def test_pyin_accuracy_440(self):
        """pYIN should detect A4 accurately on a clean sine tone."""
        sr = 48000
        audio = gen_tone(440.0, sr=sr, dur=1.0)
        result = detect_pitch_bulk(audio, sr, method="pyin")
        voiced = result.frequencies[result.frequencies > 0]
        assert len(voiced) > 0
        median_freq = np.median(voiced)
        assert abs(median_freq - 440.0) / 440.0 < 0.02

    def test_pyin_confidence_range(self):
        """pYIN confidences should be in [0, 1]."""
        sr = 48000
        audio = gen_tone(440.0, sr=sr, dur=0.5)
        result = detect_pitch_bulk(audio, sr, method="pyin")
        assert np.all(result.confidences >= 0.0)
        assert np.all(result.confidences <= 1.0)

    @pytest.mark.parametrize("freq", [220.0, 440.0, 880.0])
    def test_pyin_multiple_frequencies(self, freq):
        """pYIN should detect various frequencies accurately."""
        sr = 48000
        audio = gen_tone(freq, sr=sr, dur=1.0)
        result = detect_pitch_bulk(audio, sr, method="pyin")
        voiced = result.frequencies[result.frequencies > 0]
        if len(voiced) > 0:
            median_freq = np.median(voiced)
            assert abs(median_freq - freq) / freq < 0.03


# ---------------------------------------------------------------------------
# Gated ML backend tests (skip if deps not installed)
# ---------------------------------------------------------------------------

class TestBulkCrepe:
    @pytest.fixture(autouse=True)
    def _skip_if_no_crepe(self):
        pytest.importorskip("torchcrepe")

    def test_crepe_returns_result(self):
        sr = 48000
        audio = gen_tone(440.0, sr=sr, dur=0.5)
        result = detect_pitch_bulk(audio, sr, method="crepe", crepe_model="tiny")
        assert isinstance(result, BulkPitchResult)
        assert len(result.times) > 0


class TestBulkPesto:
    @pytest.fixture(autouse=True)
    def _skip_if_no_pesto(self):
        pytest.importorskip("pesto")

    def test_pesto_returns_result(self):
        sr = 48000
        audio = gen_tone(440.0, sr=sr, dur=0.5)
        result = detect_pitch_bulk(audio, sr, method="pesto")
        assert isinstance(result, BulkPitchResult)
        assert len(result.times) > 0


class TestPolyphonicBasicPitch:
    @pytest.fixture(autouse=True)
    def _skip_if_no_basic_pitch(self):
        pytest.importorskip("basic_pitch")

    def test_polyphonic_detects_a4(self, tmp_path):
        import soundfile as sf

        sr = 44100
        audio = gen_tone(440.0, sr=sr, dur=1.0)
        wav_path = tmp_path / "a4.wav"
        sf.write(str(wav_path), audio, sr)

        result = detect_pitch_polyphonic(str(wav_path))
        assert isinstance(result, PolyphonicResult)
        assert len(result.note_events) > 0
        # At least one event should be near A4
        notes = [e.note for e in result.note_events]
        assert any("A" in n for n in notes)


# ---------------------------------------------------------------------------
# TestImportGating
# ---------------------------------------------------------------------------

class TestImportGating:
    def test_crepe_import_error_message(self):
        """Missing torchcrepe should raise ImportError with install hint."""
        try:
            import torchcrepe  # noqa: F401
            pytest.skip("torchcrepe is installed")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="pip install mellymell"):
            detect_pitch_bulk(np.zeros(4096), 48000, method="crepe")

    def test_pesto_import_error_message(self):
        """Missing pesto should raise ImportError with install hint."""
        try:
            import pesto  # noqa: F401
            pytest.skip("pesto is installed")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="pip install mellymell"):
            detect_pitch_bulk(np.zeros(4096), 48000, method="pesto")

    def test_basic_pitch_import_error_message(self):
        """Missing basic_pitch should raise ImportError with install hint."""
        try:
            from basic_pitch import inference  # noqa: F401
            pytest.skip("basic_pitch is installed")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="pip install mellymell"):
            detect_pitch_polyphonic("/nonexistent.wav")


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            detect_pitch_bulk(np.zeros(4096), 48000, method="nonexistent")

    def test_negative_sr_raises(self):
        with pytest.raises(ValueError, match="Sample rate"):
            detect_pitch_bulk(np.zeros(4096), -1, method="yin")

    def test_zero_sr_raises(self):
        with pytest.raises(ValueError, match="Sample rate"):
            detect_pitch_bulk(np.zeros(4096), 0, method="yin")


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_bulk_pitch_result(self):
        r = BulkPitchResult(
            times=np.array([0.0, 0.1]),
            frequencies=np.array([440.0, 441.0]),
            confidences=np.array([0.9, 0.8]),
        )
        assert len(r.times) == 2
        assert r.frequencies[0] == 440.0

    def test_note_event(self):
        e = NoteEvent(
            start_s=0.0, end_s=1.0, midi_pitch=69,
            amplitude=0.8, frequency=440.0, note="A4",
        )
        assert e.midi_pitch == 69
        assert e.note == "A4"

    def test_polyphonic_result(self):
        r = PolyphonicResult(note_events=[])
        assert len(r.note_events) == 0
