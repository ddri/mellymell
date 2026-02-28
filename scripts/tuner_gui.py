#!/usr/bin/env python3
"""
GUI-based realtime tuner for mellymell using tkinter.
"""
from __future__ import annotations

import argparse
import collections
import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Deque, Optional

import numpy as np
import sounddevice as sd

from mellymell.pitch import detect_pitch, hz_to_note, note_to_hz


def parse_note_string(note_str: str) -> tuple[str, int]:
    """Parse a note string like 'A4' or 'C#3' into (name, octave)."""
    # Note names are 1-2 chars (e.g. 'A', 'C#'), octave is the trailing integer
    for i in range(len(note_str) - 1, 0, -1):
        if not note_str[i].lstrip('-').isdigit():
            return note_str[:i + 1], int(note_str[i + 1:])
    raise ValueError(f"Cannot parse note string: {note_str!r}")


class TunerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("mellymell - Realtime Tuner")
        self.root.geometry("500x400")
        self.root.configure(bg="#2b2b2b")
        
        # Audio parameters
        self.samplerate = 48000
        self.block_size = 2048
        self.tuning = 440.0
        self.fmin = 50.0
        self.fmax = 2000.0
        self.conf_threshold = 0.2
        self.median_window = 5
        self.hysteresis = 10.0
        self.method = "yin"  # Default algorithm
        
        # Audio processing state
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.freq_hist: Deque[float] = collections.deque(maxlen=self.median_window)
        self.shown_note: Optional[str] = None
        self.shown_freq: Optional[float] = None
        self.running = False
        
        self.setup_ui()
        self.setup_audio()
        
    def setup_ui(self):
        """Create the GUI layout."""
        # Main frame
        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Note display (large, center)
        self.note_label = tk.Label(
            main_frame,
            text="--",
            font=("Helvetica", 72, "bold"),
            fg="#00ff88",
            bg="#2b2b2b"
        )
        self.note_label.pack(pady=20)
        
        # Cents deviation bar frame
        cents_frame = tk.Frame(main_frame, bg="#2b2b2b")
        cents_frame.pack(pady=10)
        
        # Cents labels
        tk.Label(cents_frame, text="-50¢", fg="#888", bg="#2b2b2b").pack(side="left")
        
        # Cents meter canvas
        self.cents_canvas = tk.Canvas(
            cents_frame,
            width=300,
            height=40,
            bg="#1a1a1a",
            highlightthickness=0
        )
        self.cents_canvas.pack(side="left", padx=10)
        
        tk.Label(cents_frame, text="+50¢", fg="#888", bg="#2b2b2b").pack(side="right")
        
        # Initialize cents meter graphics
        self.setup_cents_meter()
        
        # Info display frame
        info_frame = tk.Frame(main_frame, bg="#2b2b2b")
        info_frame.pack(pady=20)
        
        self.freq_label = tk.Label(
            info_frame,
            text="--- Hz",
            font=("Helvetica", 16),
            fg="#ffaa00",
            bg="#2b2b2b"
        )
        self.freq_label.pack()
        
        self.conf_label = tk.Label(
            info_frame,
            text="Confidence: --",
            font=("Helvetica", 12),
            fg="#888",
            bg="#2b2b2b"
        )
        self.conf_label.pack()
        
        # Controls frame
        controls_frame = tk.Frame(main_frame, bg="#2b2b2b")
        controls_frame.pack(pady=20, fill="x")
        
        # Device selection
        tk.Label(controls_frame, text="Device:", fg="#ccc", bg="#2b2b2b").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.device_var,
            state="readonly",
            width=30
        )
        self.device_combo.pack(side="left", padx=5)
        self.populate_devices()
        
        # Start/Stop button
        self.start_button = tk.Button(
            controls_frame,
            text="Start",
            command=self.toggle_audio,
            bg="#00aa44",
            fg="white",
            font=("Helvetica", 12, "bold")
        )
        self.start_button.pack(side="right", padx=10)
        
        # Settings frame (expandable)
        settings_frame = tk.LabelFrame(
            main_frame,
            text="Settings",
            fg="#ccc",
            bg="#2b2b2b",
            font=("Helvetica", 10)
        )
        settings_frame.pack(pady=10, fill="x")
        
        # Tuning setting
        tuning_frame = tk.Frame(settings_frame, bg="#2b2b2b")
        tuning_frame.pack(anchor="w", padx=5, pady=2)
        tk.Label(tuning_frame, text="A4 Tuning:", fg="#ccc", bg="#2b2b2b", width=12, anchor="w").pack(side="left")
        self.tuning_var = tk.StringVar(value=str(self.tuning))
        tuning_entry = tk.Entry(tuning_frame, textvariable=self.tuning_var, width=8)
        tuning_entry.pack(side="left", padx=5)
        tk.Label(tuning_frame, text="Hz", fg="#ccc", bg="#2b2b2b").pack(side="left")
        
        # Confidence threshold
        conf_frame = tk.Frame(settings_frame, bg="#2b2b2b")
        conf_frame.pack(anchor="w", padx=5, pady=2)
        tk.Label(conf_frame, text="Min Confidence:", fg="#ccc", bg="#2b2b2b", width=12, anchor="w").pack(side="left")
        self.conf_var = tk.StringVar(value=str(self.conf_threshold))
        conf_entry = tk.Entry(conf_frame, textvariable=self.conf_var, width=8)
        conf_entry.pack(side="left", padx=5)
        
        # Algorithm selection
        algo_frame = tk.Frame(settings_frame, bg="#2b2b2b")
        algo_frame.pack(anchor="w", padx=5, pady=2)
        tk.Label(algo_frame, text="Algorithm:", fg="#ccc", bg="#2b2b2b", width=12, anchor="w").pack(side="left")
        self.method_var = tk.StringVar(value=self.method)
        method_combo = ttk.Combobox(
            algo_frame,
            textvariable=self.method_var,
            values=["yin", "mpm"],
            state="readonly",
            width=8
        )
        method_combo.pack(side="left", padx=5)
        
    def setup_cents_meter(self):
        """Initialize the cents deviation meter graphics."""
        canvas = self.cents_canvas
        w, h = 300, 40
        
        # Background
        canvas.create_rectangle(0, 0, w, h, fill="#1a1a1a", outline="#444")
        
        # Center line
        center_x = w // 2
        canvas.create_line(center_x, 0, center_x, h, fill="#666", width=2)
        
        # Tick marks every 10 cents
        for cents in range(-50, 51, 10):
            if cents == 0:
                continue
            x = center_x + (cents / 50.0) * (w // 2 - 10)
            canvas.create_line(x, h-8, x, h, fill="#444")
        
        # Store meter elements for updates
        self.cents_needle = None
        self.cents_value_text = None
        
    def update_cents_meter(self, cents: float):
        """Update the cents deviation meter."""
        canvas = self.cents_canvas
        w, h = 300, 40
        center_x = w // 2
        
        # Remove old needle and text
        if self.cents_needle:
            canvas.delete(self.cents_needle)
        if self.cents_value_text:
            canvas.delete(self.cents_value_text)
        
        # Clamp cents to display range
        cents_clamped = max(-50, min(50, cents))
        needle_x = center_x + (cents_clamped / 50.0) * (w // 2 - 10)
        
        # Color based on deviation
        if abs(cents) <= 5:
            color = "#00ff88"  # Green - in tune
        elif abs(cents) <= 15:
            color = "#ffaa00"  # Orange - close
        else:
            color = "#ff4444"  # Red - out of tune
        
        # Draw needle
        self.cents_needle = canvas.create_oval(
            needle_x - 6, h//2 - 6,
            needle_x + 6, h//2 + 6,
            fill=color, outline="white", width=2
        )
        
        # Draw cents value
        self.cents_value_text = canvas.create_text(
            center_x, 8,
            text=f"{cents:+.1f}¢",
            fill=color,
            font=("Helvetica", 10, "bold")
        )
    
    def populate_devices(self):
        """Populate the audio device dropdown."""
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                name = f"{i}: {device['name']}"
                input_devices.append(name)
        
        self.device_combo['values'] = input_devices
        if input_devices:
            self.device_combo.current(0)
    
    def setup_audio(self):
        """Setup audio processing."""
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            self.audio_queue.put(indata.copy().reshape(frames, -1))
        
        try:
            device_str = self.device_var.get()
            device_id = int(device_str.split(':')[0]) if device_str else None
            
            self.stream = sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=self.samplerate,
                blocksize=self.block_size,
                callback=audio_callback
            )
        except Exception as e:
            print(f"Audio setup error: {e}")
    
    def toggle_audio(self):
        """Start or stop audio processing."""
        if not self.running:
            self.start_audio()
        else:
            self.stop_audio()
    
    def start_audio(self):
        """Start audio processing."""
        try:
            # Update parameters from GUI
            self.tuning = float(self.tuning_var.get())
            self.conf_threshold = float(self.conf_var.get())
            self.method = self.method_var.get()
        except ValueError:
            pass  # Keep current values if invalid
        
        # Setup audio stream
        self.setup_audio()
        
        if self.stream:
            self.stream.start()
            self.running = True
            self.start_button.config(text="Stop", bg="#aa0000")
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
            self.processing_thread.start()
    
    def stop_audio(self):
        """Stop audio processing."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.start_button.config(text="Start", bg="#00aa44")
        
        # Reset display
        self.note_label.config(text="--", fg="#00ff88")
        self.freq_label.config(text="--- Hz")
        self.conf_label.config(text="Confidence: --")
        if self.cents_needle:
            self.cents_canvas.delete(self.cents_needle)
        if self.cents_value_text:
            self.cents_canvas.delete(self.cents_value_text)
    
    def process_audio(self):
        """Main audio processing loop (runs in separate thread)."""
        while self.running:
            try:
                # Get audio block with timeout
                block = self.audio_queue.get(timeout=0.1)
                block = block.squeeze(-1).astype(np.float32)
                
                # Detect pitch
                res = detect_pitch(block, self.samplerate, fmin=self.fmin, fmax=self.fmax, method=self.method)
                f = float(res.frequency)
                conf = float(res.confidence)
                
                # Update GUI in main thread
                self.root.after(0, self.update_display, f, conf)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                break
    
    def update_display(self, frequency: float, confidence: float):
        """Update the GUI display (called from main thread)."""
        # Gate low-confidence frames
        if not (np.isfinite(frequency) and frequency > 0 and confidence >= self.conf_threshold):
            self.note_label.config(text="--", fg="#666")
            self.freq_label.config(text="--- Hz")
            self.conf_label.config(text="Confidence: --")
            return
        
        # Rolling median smoothing
        self.freq_hist.append(frequency)
        f_med = float(np.median(self.freq_hist)) if len(self.freq_hist) > 0 else frequency
        name, octave, cents = hz_to_note(f_med, a4=self.tuning)
        cur_note = f"{name}{octave}"
        
        # Hysteresis on note changes
        if self.shown_note is not None and cur_note != self.shown_note:
            try:
                prev_name, prev_octave = parse_note_string(self.shown_note)
                f_prev = note_to_hz(prev_name, prev_octave, a4=self.tuning)
                cents_delta = 1200.0 * np.log2(f_med / f_prev)
                if abs(cents_delta) < self.hysteresis:
                    cur_note = self.shown_note
            except (ValueError, IndexError):
                pass  # Use new note if parsing fails
        
        self.shown_note = cur_note
        self.shown_freq = f_med
        
        # Update displays
        self.note_label.config(text=cur_note, fg="#00ff88")
        self.freq_label.config(text=f"{f_med:.2f} Hz")
        self.conf_label.config(text=f"Confidence: {confidence:.2f}")
        self.update_cents_meter(cents)
    
    def on_closing(self):
        """Handle window closing."""
        self.stop_audio()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="GUI-based realtime tuner")
    parser.add_argument("--samplerate", type=int, default=48000)
    parser.add_argument("--block", type=int, default=2048)
    parser.add_argument("--tuning", type=float, default=440.0)
    args = parser.parse_args()
    
    root = tk.Tk()
    tuner = TunerGUI(root)
    
    # Apply command line arguments
    tuner.samplerate = args.samplerate
    tuner.block_size = args.block
    tuner.tuning = args.tuning
    tuner.tuning_var.set(str(args.tuning))
    
    root.protocol("WM_DELETE_WINDOW", tuner.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()