import numpy as np
import pytest

from mellymell.pitch import detect_pitch


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


@pytest.mark.parametrize("freq", [220.0, 440.0, 880.0])
@pytest.mark.parametrize("wave", ["sine", "triangle"])
def test_detect_pitch_tones(freq, wave):
    sr = 48000
    y = gen_tone(freq, sr=sr, dur=0.1, wave=wave)
    res = detect_pitch(y, sr)
    assert res.confidence >= 0.2
    assert abs(res.frequency - freq) / freq < 0.02  # within 2%

