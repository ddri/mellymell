#!/usr/bin/env python3
"""
Comprehensive benchmarks for mellymell pitch detection accuracy and performance.
"""
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from mellymell.pitch import detect_pitch, hz_to_note


@dataclass
class BenchmarkResult:
    """Result of a single pitch detection benchmark test."""
    test_name: str
    target_freq: float
    detected_freq: float
    confidence: float
    frequency_error_hz: float
    frequency_error_cents: float
    processing_time_ms: float
    passed: bool


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results with statistics."""
    results: List[BenchmarkResult]
    total_tests: int
    passed_tests: int
    pass_rate: float
    mean_error_cents: float
    median_error_cents: float
    max_error_cents: float
    mean_processing_time_ms: float


class ToneGenerator:
    """Advanced synthetic tone generator for benchmarking."""
    
    @staticmethod
    def sine_wave(freq: float, sr: int = 48000, duration: float = 0.1, amplitude: float = 0.5) -> np.ndarray:
        """Generate pure sine wave."""
        t = np.arange(int(sr * duration)) / sr
        return amplitude * np.sin(2 * np.pi * freq * t)
    
    @staticmethod
    def sawtooth_wave(freq: float, sr: int = 48000, duration: float = 0.1, amplitude: float = 0.5) -> np.ndarray:
        """Generate band-limited sawtooth wave."""
        t = np.arange(int(sr * duration)) / sr
        y = np.zeros_like(t)
        # Add harmonics up to Nyquist
        max_harmonic = int(sr / (2 * freq))
        for k in range(1, min(50, max_harmonic)):  # Limit harmonics for efficiency
            y += (amplitude / k) * np.sin(2 * np.pi * k * freq * t)
        return y * (2 / np.pi)
    
    @staticmethod
    def square_wave(freq: float, sr: int = 48000, duration: float = 0.1, amplitude: float = 0.5) -> np.ndarray:
        """Generate band-limited square wave."""
        t = np.arange(int(sr * duration)) / sr
        y = np.zeros_like(t)
        # Add odd harmonics only
        max_harmonic = int(sr / (2 * freq))
        for k in range(1, min(25, max_harmonic), 2):  # Odd harmonics only
            y += (amplitude / k) * np.sin(2 * np.pi * k * freq * t)
        return y * (4 / np.pi)
    
    @staticmethod
    def triangle_wave(freq: float, sr: int = 48000, duration: float = 0.1, amplitude: float = 0.5) -> np.ndarray:
        """Generate triangle wave."""
        t = np.arange(int(sr * duration)) / sr
        return amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * freq * t))
    
    @staticmethod
    def vibrato_tone(freq: float, vibrato_rate: float = 5.0, vibrato_depth: float = 0.02,
                    sr: int = 48000, duration: float = 0.5, amplitude: float = 0.5) -> np.ndarray:
        """Generate tone with vibrato (frequency modulation)."""
        t = np.arange(int(sr * duration)) / sr
        vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        instantaneous_freq = freq * (1 + vibrato)
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sr
        return amplitude * np.sin(phase)
    
    @staticmethod
    def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add white noise to signal at specified SNR."""
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        return signal + noise


class PitchBenchmarker:
    """Comprehensive pitch detection benchmarking suite."""
    
    def __init__(self, sr: int = 48000):
        self.sr = sr
        self.results: List[BenchmarkResult] = []
    
    def hz_to_cents_error(self, detected: float, target: float) -> float:
        """Calculate frequency error in cents."""
        if detected <= 0 or target <= 0:
            return float('inf')
        return 1200 * np.log2(detected / target)
    
    def run_single_test(self, test_name: str, signal: np.ndarray, target_freq: float,
                       tolerance_cents: float = 50.0, method: str = "yin") -> BenchmarkResult:
        """Run a single pitch detection test."""
        start_time = time.perf_counter()
        result = detect_pitch(signal, self.sr, method=method)
        end_time = time.perf_counter()
        
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        error_hz = result.frequency - target_freq
        error_cents = self.hz_to_cents_error(result.frequency, target_freq)
        passed = abs(error_cents) <= tolerance_cents and result.confidence >= 0.2
        
        return BenchmarkResult(
            test_name=f"{method}_{test_name}",
            target_freq=target_freq,
            detected_freq=result.frequency,
            confidence=result.confidence,
            frequency_error_hz=error_hz,
            frequency_error_cents=error_cents,
            processing_time_ms=processing_time,
            passed=passed
        )
    
    def test_frequency_range(self, waveform: str = "sine", tolerance_cents: float = 30.0, 
                            algorithms: List[str] = None):
        """Test accuracy across the full frequency range."""
        if algorithms is None:
            algorithms = ["yin", "mpm"]
            
        # Musical frequencies from C1 to C8
        frequencies = []
        
        # Chromatic scale frequencies
        for octave in range(1, 9):  # C1 to C8
            for semitone in range(12):
                # A4 = 440 Hz is MIDI note 69
                midi_note = (octave + 1) * 12 + semitone
                freq = 440.0 * (2 ** ((midi_note - 69) / 12))
                if 50 <= freq <= 2000:  # Within algorithm range
                    frequencies.append(freq)
        
        generator_func = getattr(ToneGenerator, f"{waveform}_wave")
        
        for algorithm in algorithms:
            for freq in frequencies:
                signal = generator_func(freq, self.sr)
                test_name = f"freq_range_{waveform}_{freq:.1f}hz"
                result = self.run_single_test(test_name, signal, freq, tolerance_cents, method=algorithm)
                self.results.append(result)
    
    def test_waveforms(self, test_frequencies: List[float] = None, algorithms: List[str] = None):
        """Test different waveforms at standard frequencies."""
        if test_frequencies is None:
            test_frequencies = [110.0, 220.0, 440.0, 880.0]  # A2, A3, A4, A5
        if algorithms is None:
            algorithms = ["yin", "mpm"]
        
        waveforms = ["sine", "sawtooth", "square", "triangle"]
        
        for algorithm in algorithms:
            for waveform in waveforms:
                generator_func = getattr(ToneGenerator, f"{waveform}_wave")
                for freq in test_frequencies:
                    signal = generator_func(freq, self.sr)
                    test_name = f"waveform_{waveform}_{freq:.1f}hz"
                    result = self.run_single_test(test_name, signal, freq, method=algorithm)
                    self.results.append(result)
    
    def test_noise_robustness(self, test_frequencies: List[float] = None, 
                             snr_levels: List[float] = None):
        """Test performance with varying noise levels."""
        if test_frequencies is None:
            test_frequencies = [220.0, 440.0, 880.0]
        if snr_levels is None:
            snr_levels = [40, 30, 20, 15, 10, 5]  # dB
        
        for freq in test_frequencies:
            for snr in snr_levels:
                clean_signal = ToneGenerator.sine_wave(freq, self.sr)
                noisy_signal = ToneGenerator.add_noise(clean_signal, snr)
                test_name = f"noise_robustness_{freq:.1f}hz_snr{snr}db"
                
                # More lenient tolerance for noisy signals
                tolerance = 50.0 if snr >= 20 else 100.0
                result = self.run_single_test(test_name, noisy_signal, freq, tolerance)
                self.results.append(result)
    
    def test_vibrato_tracking(self, test_frequencies: List[float] = None):
        """Test pitch tracking with vibrato/frequency modulation."""
        if test_frequencies is None:
            test_frequencies = [220.0, 440.0, 880.0]
        
        vibrato_rates = [3.0, 5.0, 7.0]  # Hz
        vibrato_depths = [0.01, 0.02, 0.03]  # 1%, 2%, 3%
        
        for freq in test_frequencies:
            for rate in vibrato_rates:
                for depth in vibrato_depths:
                    signal = ToneGenerator.vibrato_tone(
                        freq, rate, depth, self.sr, duration=0.5
                    )
                    test_name = f"vibrato_{freq:.1f}hz_rate{rate}hz_depth{depth*100:.0f}pct"
                    
                    # Vibrato makes exact frequency detection harder
                    result = self.run_single_test(test_name, signal, freq, tolerance_cents=100.0)
                    self.results.append(result)
    
    def test_edge_cases(self):
        """Test edge cases and potential failure modes."""
        # Very low frequencies
        for freq in [55.0, 65.0, 82.0]:  # A1, C2, E2
            signal = ToneGenerator.sine_wave(freq, self.sr, duration=0.2)
            test_name = f"edge_low_freq_{freq:.1f}hz"
            result = self.run_single_test(test_name, signal, freq, tolerance_cents=100.0)
            self.results.append(result)
        
        # Very high frequencies
        for freq in [1760.0, 1975.0]:  # A6, B6
            signal = ToneGenerator.sine_wave(freq, self.sr, duration=0.05)
            test_name = f"edge_high_freq_{freq:.1f}hz"
            result = self.run_single_test(test_name, signal, freq, tolerance_cents=50.0)
            self.results.append(result)
        
        # Short duration signals
        for duration in [0.02, 0.03, 0.05]:  # 20ms, 30ms, 50ms
            signal = ToneGenerator.sine_wave(440.0, self.sr, duration=duration)
            test_name = f"edge_short_duration_{duration*1000:.0f}ms"
            result = self.run_single_test(test_name, signal, 440.0, tolerance_cents=100.0)
            self.results.append(result)
        
        # Very quiet signals
        for amplitude in [0.1, 0.05, 0.01]:
            signal = ToneGenerator.sine_wave(440.0, self.sr, amplitude=amplitude)
            test_name = f"edge_quiet_amplitude_{amplitude:.2f}"
            result = self.run_single_test(test_name, signal, 440.0, tolerance_cents=100.0)
            self.results.append(result)
    
    def run_performance_benchmarks(self, iterations: int = 100):
        """Benchmark processing performance."""
        # Test different frame sizes
        frame_sizes = [1024, 2048, 4096, 8192]
        test_freq = 440.0
        
        for frame_size in frame_sizes:
            durations = []
            signal = ToneGenerator.sine_wave(test_freq, self.sr, 
                                           duration=frame_size / self.sr)
            
            # Warm up
            for _ in range(10):
                detect_pitch(signal, self.sr)
            
            # Benchmark
            for _ in range(iterations):
                start = time.perf_counter()
                detect_pitch(signal, self.sr)
                end = time.perf_counter()
                durations.append((end - start) * 1000)
            
            mean_time = np.mean(durations)
            std_time = np.std(durations)
            
            result = BenchmarkResult(
                test_name=f"performance_frame{frame_size}",
                target_freq=test_freq,
                detected_freq=test_freq,  # Not relevant for perf test
                confidence=1.0,
                frequency_error_hz=0.0,
                frequency_error_cents=0.0,
                processing_time_ms=mean_time,
                passed=True
            )
            self.results.append(result)
            print(f"Frame size {frame_size}: {mean_time:.2f}±{std_time:.2f}ms")
    
    def generate_report(self) -> BenchmarkSuite:
        """Generate comprehensive benchmark report."""
        if not self.results:
            raise ValueError("No benchmark results available")
        
        passed_results = [r for r in self.results if r.passed]
        valid_errors = [r.frequency_error_cents for r in self.results 
                       if np.isfinite(r.frequency_error_cents)]
        
        return BenchmarkSuite(
            results=self.results,
            total_tests=len(self.results),
            passed_tests=len(passed_results),
            pass_rate=len(passed_results) / len(self.results) * 100,
            mean_error_cents=float(np.mean(valid_errors)) if valid_errors else float('inf'),
            median_error_cents=float(np.median(valid_errors)) if valid_errors else float('inf'),
            max_error_cents=float(np.max(np.abs(valid_errors))) if valid_errors else float('inf'),
            mean_processing_time_ms=float(np.mean([r.processing_time_ms for r in self.results]))
        )
    
    def save_report(self, filepath: str):
        """Save benchmark report to JSON file."""
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(asdict(report), f, indent=2)
    
    def run_full_suite(self, save_path: Optional[str] = None):
        """Run the complete benchmark suite."""
        print("🚀 Running comprehensive mellymell benchmarks...")
        
        print("📊 Testing frequency range accuracy...")
        self.test_frequency_range("sine", tolerance_cents=30.0)
        
        print("🎵 Testing different waveforms...")
        self.test_waveforms()
        
        print("🔊 Testing noise robustness...")
        self.test_noise_robustness()
        
        print("🎼 Testing vibrato tracking...")
        self.test_vibrato_tracking()
        
        print("⚠️  Testing edge cases...")
        self.test_edge_cases()
        
        print("⚡ Running performance benchmarks...")
        self.run_performance_benchmarks(iterations=50)
        
        report = self.generate_report()
        
        print(f"\n📈 BENCHMARK RESULTS:")
        print(f"Total tests: {report.total_tests}")
        print(f"Passed: {report.passed_tests} ({report.pass_rate:.1f}%)")
        print(f"Mean error: {report.mean_error_cents:.1f} cents")
        print(f"Median error: {report.median_error_cents:.1f} cents")
        print(f"Max error: {report.max_error_cents:.1f} cents")
        print(f"Mean processing time: {report.mean_processing_time_ms:.2f} ms")
        
        if save_path:
            self.save_report(save_path)
            print(f"\n💾 Report saved to: {save_path}")
        
        return report


# Pytest integration
def test_basic_accuracy():
    """Basic accuracy test for CI."""
    benchmarker = PitchBenchmarker()
    benchmarker.test_waveforms([440.0])  # Just test A4
    
    results = [r for r in benchmarker.results if r.target_freq == 440.0]
    passed_results = [r for r in results if r.passed]
    
    assert len(passed_results) >= 3, f"Expected at least 3/4 waveforms to pass, got {len(passed_results)}"
    
    # Check that sine wave is very accurate
    sine_result = next((r for r in results if "sine" in r.test_name), None)
    assert sine_result is not None
    assert abs(sine_result.frequency_error_cents) < 10.0, f"Sine wave error too high: {sine_result.frequency_error_cents:.1f} cents"


if __name__ == "__main__":
    # Run full benchmark suite when executed directly
    benchmarker = PitchBenchmarker()
    report = benchmarker.run_full_suite("benchmark_report.json")