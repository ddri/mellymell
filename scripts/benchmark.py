#!/usr/bin/env python3
"""
CLI tool for running mellymell benchmarks.
"""
import argparse
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import from tests
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_benchmarks import PitchBenchmarker


def main():
    parser = argparse.ArgumentParser(description="Run mellymell pitch detection benchmarks")
    parser.add_argument(
        "--output", "-o", 
        default="benchmark_report.json",
        help="Output file for JSON report"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick benchmark suite (basic tests only)"
    )
    parser.add_argument(
        "--performance-only", "-p",
        action="store_true", 
        help="Run performance benchmarks only"
    )
    parser.add_argument(
        "--accuracy-only", "-a",
        action="store_true",
        help="Run accuracy benchmarks only"
    )
    parser.add_argument(
        "--algorithms", 
        nargs="+",
        default=["yin", "mpm"],
        choices=["yin", "mpm"],
        help="Algorithms to benchmark (default: both yin and mpm)"
    )
    
    args = parser.parse_args()
    
    benchmarker = PitchBenchmarker()
    
    if args.quick:
        print("🚀 Running quick benchmark suite...")
        benchmarker.test_waveforms([220.0, 440.0, 880.0], algorithms=args.algorithms)
        benchmarker.test_frequency_range("sine", tolerance_cents=50.0, algorithms=args.algorithms)
        
    elif args.performance_only:
        print("⚡ Running performance benchmarks only...")
        benchmarker.run_performance_benchmarks(iterations=100)
        
    elif args.accuracy_only:
        print("🎯 Running accuracy benchmarks only...")
        benchmarker.test_frequency_range("sine", algorithms=args.algorithms)
        benchmarker.test_waveforms(algorithms=args.algorithms)
        benchmarker.test_noise_robustness()
        benchmarker.test_vibrato_tracking()
        benchmarker.test_edge_cases()
        
    else:
        # Full suite
        benchmarker.run_full_suite()
    
    # Generate and save report
    report = benchmarker.generate_report()
    benchmarker.save_report(args.output)
    
    print(f"\n📈 FINAL SUMMARY:")
    print(f"Pass rate: {report.pass_rate:.1f}% ({report.passed_tests}/{report.total_tests})")
    print(f"Median error: {report.median_error_cents:.1f} cents")
    print(f"Mean processing: {report.mean_processing_time_ms:.2f} ms/frame")
    print(f"Report saved: {args.output}")
    
    # Exit with error code if pass rate is too low
    if report.pass_rate < 70.0:
        print(f"\n❌ BENCHMARK FAILED: Pass rate {report.pass_rate:.1f}% below 70% threshold")
        sys.exit(1)
    else:
        print(f"\n✅ BENCHMARKS PASSED")


if __name__ == "__main__":
    main()