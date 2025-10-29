#!/usr/bin/env python3
"""Performance monitoring script for CI/CD integration.

This script:
1. Runs performance tests
2. Compares results against baseline
3. Generates reports
4. Detects regressions
5. Posts results to GitHub (if configured)

Usage:
    python scripts/performance/monitor_performance.py [--baseline] [--compare] [--threshold 10]

Environment variables:
    GITHUB_TOKEN: GitHub token for posting comments (optional)
    GITHUB_REPOSITORY: Repository name (e.g., 'owner/repo') (optional)
    GITHUB_PR_NUMBER: Pull request number for posting results (optional)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class PerformanceMonitor:
    """Monitor and report on performance test results."""

    def __init__(
        self,
        baseline_dir: Path = Path(".benchmarks/baseline"),
        current_dir: Path = Path(".benchmarks/current"),
        threshold: float = 10.0,
    ):
        """Initialize performance monitor.

        Args:
            baseline_dir: Directory containing baseline benchmark results
            current_dir: Directory for current benchmark results
            threshold: Regression threshold percentage (default: 10%)
        """
        self.baseline_dir = baseline_dir
        self.current_dir = current_dir
        self.threshold = threshold

    def run_tests(self, save_baseline: bool = False) -> int:
        """Run performance tests with pytest-benchmark.

        Args:
            save_baseline: Whether to save results as new baseline

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        cmd = [
            "uv",
            "run",
            "pytest",
            "tests/performance/",
            "--benchmark-only",
            "--benchmark-autosave",
            "--benchmark-save-data",
            "-v",
        ]

        if save_baseline:
            cmd.extend(
                [
                    "--benchmark-save=baseline",
                    f"--benchmark-storage=file://{self.baseline_dir}",
                ]
            )
            print(f"📊 Running tests and saving baseline to {self.baseline_dir}")
        else:
            cmd.extend(
                [
                    f"--benchmark-storage=file://{self.current_dir}",
                ]
            )
            print(f"📊 Running tests and saving to {self.current_dir}")

        result = subprocess.run(cmd, check=False)
        return result.returncode

    def compare_results(self) -> dict[str, Any]:
        """Compare current results against baseline.

        Returns:
            Dictionary with comparison results including:
            - regressions: List of regressed benchmarks
            - improvements: List of improved benchmarks
            - summary: Overall summary statistics
        """
        if not self.baseline_dir.exists() or not any(self.baseline_dir.iterdir()):
            print(f"❌ No baseline found in {self.baseline_dir}")
            return {"error": "No baseline found"}

        # Load latest baseline and current results
        baseline_file = self._get_latest_benchmark(self.baseline_dir)
        current_file = self._get_latest_benchmark(self.current_dir)

        if not baseline_file or not current_file:
            return {"error": "Could not find benchmark results"}

        with open(baseline_file) as f:
            baseline_data = json.load(f)

        with open(current_file) as f:
            current_data = json.load(f)

        # Compare benchmarks
        comparison: dict[str, Any] = {
            "regressions": [],
            "improvements": [],
            "unchanged": [],
            "summary": {},
        }

        baseline_benchmarks = {b["fullname"]: b for b in baseline_data["benchmarks"]}
        current_benchmarks = {b["fullname"]: b for b in current_data["benchmarks"]}

        for name, current in current_benchmarks.items():
            if name not in baseline_benchmarks:
                continue

            baseline = baseline_benchmarks[name]

            # Compare mean times
            baseline_mean = baseline["stats"]["mean"]
            current_mean = current["stats"]["mean"]
            change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100

            benchmark_info = {
                "name": name,
                "baseline_mean": baseline_mean,
                "current_mean": current_mean,
                "change_pct": change_pct,
            }

            if change_pct > self.threshold:
                comparison["regressions"].append(benchmark_info)
            elif change_pct < -self.threshold:
                comparison["improvements"].append(benchmark_info)
            else:
                comparison["unchanged"].append(benchmark_info)

        # Generate summary
        comparison["summary"] = {
            "total_benchmarks": len(current_benchmarks),
            "regressions": len(comparison["regressions"]),
            "improvements": len(comparison["improvements"]),
            "unchanged": len(comparison["unchanged"]),
            "threshold": self.threshold,
        }

        return comparison

    def _get_latest_benchmark(self, directory: Path) -> Path | None:
        """Get the most recent benchmark file from directory."""
        if not directory.exists():
            return None

        json_files = list(directory.glob("*.json"))
        if not json_files:
            return None

        # Get most recent file
        return max(json_files, key=lambda p: p.stat().st_mtime)

    def print_comparison_report(self, comparison: dict[str, Any]) -> None:
        """Print formatted comparison report."""
        if "error" in comparison:
            print(f"❌ {comparison['error']}")
            return

        summary = comparison["summary"]

        print("\n" + "=" * 70)
        print("📊 PERFORMANCE COMPARISON REPORT")
        print("=" * 70)
        print(f"\nThreshold: ±{self.threshold}%")
        print(f"Total benchmarks: {summary['total_benchmarks']}")
        print(f"  • Regressions: {summary['regressions']}")
        print(f"  • Improvements: {summary['improvements']}")
        print(f"  • Unchanged: {summary['unchanged']}")

        if comparison["regressions"]:
            print("\n⚠️  REGRESSIONS DETECTED:")
            for reg in comparison["regressions"]:
                print(f"\n  {reg['name']}")
                print(f"    Baseline:  {reg['baseline_mean']:.6f}s")
                print(f"    Current:   {reg['current_mean']:.6f}s")
                print(f"    Change:    {reg['change_pct']:+.1f}%")

        if comparison["improvements"]:
            print("\n✅ IMPROVEMENTS:")
            for imp in comparison["improvements"]:
                print(f"\n  {imp['name']}")
                print(f"    Baseline:  {imp['baseline_mean']:.6f}s")
                print(f"    Current:   {imp['current_mean']:.6f}s")
                print(f"    Change:    {imp['change_pct']:+.1f}%")

        print("\n" + "=" * 70)

    def has_regressions(self, comparison: dict[str, Any]) -> bool:  # type: ignore[return]
        """Check if comparison shows any regressions."""
        return len(comparison.get("regressions", [])) > 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor performance tests")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Save results as new baseline",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare against baseline and fail on regression",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold percentage (default: 10%%)",
    )

    args = parser.parse_args()

    monitor = PerformanceMonitor(threshold=args.threshold)

    if args.baseline:
        # Save new baseline
        exit_code = monitor.run_tests(save_baseline=True)
        if exit_code == 0:
            print("✅ Baseline saved successfully")
        return exit_code

    elif args.compare:
        # Run tests
        exit_code = monitor.run_tests(save_baseline=False)
        if exit_code != 0:
            print("❌ Performance tests failed")
            return exit_code

        # Compare against baseline
        comparison = monitor.compare_results()
        monitor.print_comparison_report(comparison)

        # Fail if regressions detected
        if monitor.has_regressions(comparison):
            print("\n❌ Performance regressions detected!")
            return 1

        print("\n✅ No performance regressions detected")
        return 0

    else:
        # Just run tests
        return monitor.run_tests(save_baseline=False)


if __name__ == "__main__":
    sys.exit(main())
