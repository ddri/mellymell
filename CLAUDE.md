# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mellymell is an open-source pitch detection tool written in Python, inspired by NoteGrabber and aimed at providing Melodyne-like visual timeline analysis of detected notes. It's a prototype that implements monophonic pitch detection using the YIN algorithm.

The project has two main modes:
- **Realtime tuner**: Live mic input showing note, cents deviation, and confidence
- **Offline analysis**: Audio file analysis producing CSV/JSON outputs and Melodyne-style visualizations

## Development Setup

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
brew install portaudio  # Required for sounddevice (macOS)
pip install -U pip
pip install -e .[dev]
```

### Running Tests
```bash
pytest -q                    # Run all tests quietly
pytest tests/test_pitch.py   # Run specific test file
pytest -v                   # Verbose output
pytest tests/test_pitch.py::test_detect_pitch_tones  # Run specific test
```

### Running Benchmarks
```bash
python scripts/benchmark.py                # Full benchmark suite
python scripts/benchmark.py --quick        # Basic accuracy tests
python scripts/benchmark.py --accuracy-only # Accuracy benchmarks only
python scripts/benchmark.py --performance-only # Performance benchmarks only
```

### Running Scripts

**Realtime tuner (mic input):**
```bash
python scripts/live_tuner.py                    # YIN algorithm (default)
python scripts/live_tuner.py --method mpm       # McLeod Pitch Method
python scripts/live_tuner.py --device "Built-in Microphone" --samplerate 48000 --block 2048 --tuning 440.0
```

**GUI tuner:**
```bash
python scripts/tuner_gui.py                     # Algorithm selectable in settings
```

**Offline file analysis:**
```bash
python scripts/analyze_file.py path/to/audio.wav --output pitch.csv --segments segments.csv --plot-segments --html report.html
```

## Architecture

### Core Modules

- **`src/mellymell/pitch.py`**: Core pitch detection algorithms
  - `detect_pitch()`: Main entry point supporting YIN and MPM algorithms
  - `yin()`: YIN algorithm implementation with cumulative mean normalized difference
  - `mpm()`: McLeod Pitch Method with normalized squared difference function
  - `hz_to_note()`, `midi_to_hz()`: Frequency/MIDI/note conversion utilities
  - `PitchResult` dataclass: Stores frequency and confidence values

- **`src/mellymell/segment.py`**: Note segmentation for offline analysis
  - `segment_notes()`: Groups framewise pitch detections into note segments
  - `NoteSegment` dataclass: Stores segment timing, note, cents deviation, and confidence
  - Handles gap detection and minimum segment duration filtering

### Key Data Flow

The architecture follows a modular pipeline design:

1. **Audio Input** → Audio frames (mono, 16-bit, various sample rates)
2. **Frame Processing** → `pitch.detect_pitch()` returns `PitchResult` per frame
3. **Note Mapping** → `pitch.hz_to_note()` converts Hz to note names + cents
4. **Segmentation** → `segment.segment_notes()` groups frames into `NoteSegment` objects
5. **Output** → CSV/JSON files, HTML reports, or realtime display

### Scripts

- **`scripts/live_tuner.py`**: Realtime audio input processing with smoothing and hysteresis
- **`scripts/analyze_file.py`**: Batch processing of audio files with CSV/JSON/HTML output generation

### Signal Processing Pipeline

1. **Input**: Audio frames (mono, windowed with Hann function)
2. **Pitch Detection**: YIN algorithm with configurable frequency range (50-2000 Hz default)
3. **Smoothing**: Median filtering and hysteresis for stable realtime display
4. **Note Mapping**: Hz → MIDI → note name + octave + cents deviation
5. **Segmentation**: Group consecutive same-note detections with gap tolerance

### Key Parameters

- **Frame sizes**: 2048 samples default (adjustable for latency vs accuracy tradeoff)
- **Hop size**: 1024 samples for offline analysis
- **Confidence threshold**: 0.2 for note segmentation
- **A4 tuning**: 440.0 Hz (configurable)
- **Frequency range**: 50-2000 Hz (monophonic focus)

## Dependencies

- **Audio I/O**: sounddevice, soundfile, librosa
- **DSP**: numpy, scipy
- **Visualization**: matplotlib
- **Testing**: pytest

## CI/Testing

The project uses GitHub Actions with macOS runners (matching the primary development platform). Tests validate pitch detection accuracy using synthetic tones with parametrized frequencies and waveforms.

### Test Structure
- **`tests/test_pitch.py`**: Core algorithm validation with synthetic tones
- **`tests/test_benchmarks.py`**: Performance and accuracy benchmark tests

### Quality Assurance
- No specific linting tools configured; follow PEP 8 conventions
- All tests must pass before merging: `pytest -q`
- Benchmarks validate algorithm accuracy across frequency ranges and waveforms