#!/usr/bin/env python3
"""
Benchmark script for measuring Buttermilk startup performance.

This script measures the time from application start to first HTTP response
to quantify startup performance improvements.
"""

import asyncio
import subprocess
import sys
import time
import requests
from pathlib import Path

# Add buttermilk to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def measure_startup_time(timeout=60):
    """Measure the time from process start to first HTTP response.
    
    Args:
        timeout: Maximum time to wait for server to start (seconds)
        
    Returns:
        tuple: (startup_time, success) where startup_time is in seconds
    """
    print("🚀 Starting Buttermilk API server...")
    
    # Ensure no server is already running
    subprocess.run(["pkill", "-f", "buttermilk.runner.cli"], 
                  capture_output=True, timeout=10)
    time.sleep(2)  # Wait for cleanup
    
    # Verify no server is running
    try:
        response = requests.get("http://localhost:8000/api/flows", timeout=1)
        if response.status_code == 200:
            print("⚠️  Server already running, killing it...")
            subprocess.run(["pkill", "-f", "buttermilk.runner.cli"], 
                          capture_output=True, timeout=10)
            time.sleep(3)
    except requests.exceptions.RequestException:
        pass  # Good, no server running
    
    # Start timing from here
    start_time = time.time()
    
    cmd = [
        sys.executable, "-m", "buttermilk.runner.cli",
        "+flows=[trans, tox_allinone]", "+run=api", "+llms=lite"
    ]
    
    print(f"📋 Command: {' '.join(cmd)}")
    
    # Start server in background
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent.parent,
        text=True
    )
    
    # Wait for server to be ready
    url = "http://localhost:8000/api/flows"
    
    try:
        # Poll the server until it responds
        for i in range(timeout * 10):  # Check every 100ms
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    end_time = time.time()
                    startup_time = end_time - start_time
                    print(f"✅ Server ready! Startup time: {startup_time:.2f} seconds")
                    
                    # Quick test to make sure it's working
                    data = response.json()
                    if 'flow_choices' in data:
                        print(f"✅ Server functional with {len(data['flow_choices'])} flows")
                    
                    return startup_time, True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(0.1)
        
        print(f"❌ Server failed to start within {timeout} seconds")
        
        # Print stderr for debugging
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stderr:
                print(f"❌ Server error output: {stderr[:500]}")
        except subprocess.TimeoutExpired:
            pass
            
        return timeout, False
        
    finally:
        # Clean up the process
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_multiple_tests(num_tests=3):
    """Run multiple startup tests and calculate statistics.
    
    Args:
        num_tests: Number of test runs to perform
        
    Returns:
        dict: Statistics about startup times
    """
    print(f"🧪 Running {num_tests} startup performance tests...")
    
    times = []
    successful = 0
    
    for i in range(num_tests):
        print(f"\n--- Test {i+1}/{num_tests} ---")
        startup_time, success = measure_startup_time()
        
        if success:
            times.append(startup_time)
            successful += 1
        
        # Wait between tests to ensure clean state
        if i < num_tests - 1:
            print("⏱️  Waiting 5 seconds between tests...")
            time.sleep(5)
    
    if times:
        stats = {
            "successful_runs": successful,
            "total_runs": num_tests,
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": sum(times) / len(times),
            "times": times
        }
        
        print(f"\n📊 Startup Performance Results:")
        print(f"   Successful runs: {successful}/{num_tests}")
        print(f"   Average time: {stats['avg_time']:.2f}s")
        print(f"   Min time: {stats['min_time']:.2f}s")
        print(f"   Max time: {stats['max_time']:.2f}s")
        print(f"   All times: {[f'{t:.2f}s' for t in times]}")
        
        return stats
    else:
        print("❌ No successful runs to analyze")
        return None


def compare_with_baseline():
    """Compare current performance with documented baseline."""
    print("📋 Comparing with baseline performance...")
    
    # Run current tests
    current_stats = run_multiple_tests(3)
    
    if current_stats:
        baseline_time = 5.0  # Baseline from issue description: 5-8 seconds
        target_time = 2.0    # Target from issue: <2 seconds
        
        avg_time = current_stats['avg_time']
        improvement = baseline_time - avg_time
        improvement_pct = (improvement / baseline_time) * 100
        
        print(f"\n🎯 Performance Analysis:")
        print(f"   Baseline: ~{baseline_time:.1f}s")
        print(f"   Current: {avg_time:.2f}s")
        print(f"   Improvement: {improvement:.2f}s ({improvement_pct:.1f}%)")
        print(f"   Target: {target_time:.1f}s")
        
        if avg_time <= target_time:
            print("✅ TARGET ACHIEVED!")
        else:
            remaining = avg_time - target_time
            print(f"🎯 Still need {remaining:.2f}s improvement to reach target")
        
        return current_stats
    
    return None


def test_specific_endpoints():
    """Test that MCP endpoints work after startup."""
    print("\n🔧 Testing MCP endpoints after startup...")
    
    startup_time, success = measure_startup_time()
    
    if not success:
        print("❌ Server failed to start, skipping endpoint tests")
        return False
    
    # Test MCP endpoints
    endpoints = [
        ("http://localhost:8000/api/flows", "Core API"),
        ("http://localhost:8000/mcp/tools", "MCP Tools")
    ]
    
    all_success = True
    
    for url, name in endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: {response.status_code}")
            else:
                print(f"❌ {name}: {response.status_code}")
                all_success = False
        except Exception as e:
            print(f"❌ {name}: {e}")
            all_success = False
    
    # Clean up
    subprocess.run(["pkill", "-f", "buttermilk.runner.cli"], 
                  capture_output=True, timeout=10)
    
    return all_success


def main():
    """Main benchmark runner."""
    print("🚀 Buttermilk Startup Performance Benchmark")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "compare"
    
    if mode == "single":
        startup_time, success = measure_startup_time()
        if success:
            print(f"Single test result: {startup_time:.2f} seconds")
        return 0 if success else 1
        
    elif mode == "multiple":
        num_tests = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        stats = run_multiple_tests(num_tests)
        return 0 if stats else 1
        
    elif mode == "endpoints":
        success = test_specific_endpoints()
        return 0 if success else 1
        
    else:  # "compare" mode (default)
        stats = compare_with_baseline()
        return 0 if stats else 1


if __name__ == "__main__":
    sys.exit(main())