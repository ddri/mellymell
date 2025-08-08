# mellymell

An open-source pitch detection tool inspired by NoteGrabber.

Goals
- Prototype in Python on macOS first (fast iteration), then consider a JUCE/C++ implementation for cross-platform plugin + standalone.
- Detect the fundamental frequency (pitch) of audio and map it to musical notes with cents deviation.
- Two modes:
  - Realtime monitoring (mic/audio input): big note + cents + confidence.
  - Offline file analysis: analyze a loaded audio file and produce a timeline of detected notes.
- License: GPL-3.0-or-later.

Why Python first?
- Rapid iteration on DSP and UI/UX ideas.
- Access to robust audio/DSP libraries (numpy, scipy, librosa, sounddevice).
- Easy validation using synthetic tones and recorded samples before porting to JUCE.

Targets (now and later)
- Now (prototype): Standalone Python scripts for realtime and offline analysis on macOS.
- Later (production): JUCE-based app that can build Standalone + VST3 (and AU on macOS). Python algorithms will be ported/refined in C++.

Core algorithm (MVP)
- Monophonic pitch detection using a variant of YIN / MPM (McLeod Pitch Method) implemented in numpy.
- DSP pipeline:
  - Optional high-pass (to reduce DC/rumble), windowing.
  - Pitch detection per frame with confidence estimate.
  - Smoothing via median filtering and hysteresis to stabilize readout.
- Mapping: Convert Hz -> note name + octave + cents offset. A4 tuning configurable (default 440 Hz).

Multi-pitch/chords (future)
- Stretch goal after MVP is solid: spectral peak picking + harmonic grouping to show note sets for polyphonic audio.
- Optional ML models later (e.g., CREPE for monophonic; melody/chord extraction research tools) if we accept heavier deps.

Repository layout
- pyproject.toml          Project metadata and dependencies
- src/mellymell/
  - __init__.py
  - pitch.py              Core pitch detection utilities (YIN/MPM, Hz<->note mapping)
- scripts/
  - live_tuner.py         Realtime mic input: shows note, cents, Hz, confidence
  - analyze_file.py       Offline file analysis: writes CSV and optional plot
- README.md               This document
- LICENSE                 GPL-3.0 license
- .gitignore              Python ignores, venv, macOS, etc.

Setup (macOS)
1) Create and activate a virtualenv
- python3 -m venv .venv
- source .venv/bin/activate

2) Install system audio dependency for realtime input
- brew install portaudio   # required by sounddevice

3) Install Python deps
- pip install -U pip
- pip install -e .         # installs from pyproject.toml

Realtime tuner (mic)
- python scripts/live_tuner.py
Options:
- --device NAME_OR_ID     Audio input device (optional)
- --samplerate 48000      Sample rate (default 48000)
- --block 2048            Block size in samples (analysis frame)
- --tuning 440.0          A4 reference

Offline file analysis
- python scripts/analyze_file.py path/to/audio.wav --output pitch.csv
Options:
- --samplerate 0          If 0, use file’s native rate (recommended)
- --hop 1024              Hop size for analysis
- --tuning 440.0          A4 reference
- --plot                  Show a quick matplotlib plot (optional)

Notes on accuracy / latency
- Smaller block sizes decrease latency but can reduce accuracy for low notes.
- For stable readout, we median-filter across recent frames; you can tune window size.

Roadmap
- Prototype stability + accuracy benchmarks with synthetic tones.
- Add a minimal GUI (optional) using textual UI or simple matplotlib real-time plot.
- Begin JUCE port: replicate pitch.py algorithm in C++ and wire up a basic UI that shows note/cents/confidence; package Standalone first.

Contributing
- GPL-3.0-or-later. PRs welcome. Please include small test audio where possible.

Credits / Inspiration
- YIN: A. de Cheveigné and H. Kawahara (2002)
- MPM: P. McLeod and G. Wyvill (2005)
- Libraries: numpy, scipy, librosa, sounddevice, soundfile

