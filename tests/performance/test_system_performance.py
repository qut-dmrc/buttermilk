"""
Comprehensive Performance and Reliability Testing Suite.

This module provides thorough performance testing for any flow configuration,
including:

- Load testing under various user patterns
- Latency and throughput measurement
- Memory and resource usage monitoring
- Scalability testing across flow configurations
- Reliability testing under stress conditions
- Performance regression detection

Designed to validate system performance characteristics across any
YAML-configured flow without hardcoded dependencies.
"""

import asyncio
import pytest
import time
import psutil
import statistics
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass, asdict
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
import concurrent.futures
import threading

from buttermilk.api.flow import create_app
from buttermilk._core.bm_init import BM
from buttermilk.runner.flowrunner import FlowRunner
from tests.utils.test_data_manager import TestDataManager, create_test_flow_config


@dataclass
class PerformanceMetrics:
    """Performance metrics for test results."""
    test_name: str
    flow_name: str
    duration: float
    memory_usage_mb: float
    cpu_usage_percent: float
    request_count: int
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    error_count: int
    concurrent_users: int


@dataclass
class LoadTestResult:
    """Complete load test results."""
    test_config: Dict[str, Any]
    performance_metrics: PerformanceMetrics
    detailed_timings: List[float]
    error_log: List[str]
    resource_timeline: List[Dict[str, float]]
    success: bool


class PerformanceMonitor:
    """Monitors system performance during tests."""
    
    def __init__(self, sampling_interval: float = 1.0):
        """Initialize with sampling interval in seconds."""
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.metrics_timeline = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.metrics_timeline = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[Dict[str, float]]:
        """Stop monitoring and return collected metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        return self.metrics_timeline
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                timestamp = time.time()
                process = psutil.Process()
                
                # Collect system metrics
                metrics = {
                    "timestamp": timestamp,
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "cpu_percent": process.cpu_percent(),
                    "memory_percent": process.memory_percent(),
                    "num_threads": process.num_threads(),
                    "num_fds": process.num_fds() if hasattr(process, 'num_fds') else 0,
                    "system_memory_percent": psutil.virtual_memory().percent,
                    "system_cpu_percent": psutil.cpu_percent()
                }
                
                self.metrics_timeline.append(metrics)
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                # Continue monitoring even if individual samples fail
                pass


class LoadTestRunner:
    """Runs comprehensive load tests for any flow configuration."""
    
    def __init__(self, app, flow_configs: Dict[str, Any]):
        """Initialize with app and flow configurations."""
        self.app = app
        self.flow_configs = flow_configs
        self.data_manager = TestDataManager()
        self.performance_monitor = PerformanceMonitor()
    
    async def run_load_test(self, test_config: Dict[str, Any]) -> LoadTestResult:
        """Run comprehensive load test with given configuration."""
        flow_name = test_config["flow_name"]
        concurrent_users = test_config["concurrent_users"]
        test_duration = test_config["duration_seconds"]
        requests_per_user = test_config.get("requests_per_user", 10)
        
        print(f"Starting load test: {concurrent_users} users, {test_duration}s, flow: {flow_name}")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        # Initialize tracking
        request_timings = []
        errors = []
        successful_requests = 0
        total_requests = 0
        
        start_time = time.time()
        
        try:
            # Create user sessions
            with TestClient(self.app) as client:
                # Generate user tasks
                user_tasks = []
                for user_id in range(concurrent_users):
                    task = asyncio.create_task(
                        self._simulate_user_load(
                            client, user_id, flow_name, test_duration, 
                            requests_per_user, request_timings, errors
                        )
                    )
                    user_tasks.append(task)
                
                # Wait for all users to complete
                user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
                
                # Process results
                for result in user_results:
                    if isinstance(result, Exception):
                        errors.append(str(result))
                    else:
                        successful_requests += result.get("successful_requests", 0)
                        total_requests += result.get("total_requests", 0)
        
        except Exception as e:
            errors.append(f"Load test execution error: {e}")
        
        finally:
            # Stop monitoring
            resource_timeline = self.performance_monitor.stop_monitoring()
        
        # Calculate metrics
        total_duration = time.time() - start_time
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            test_config, request_timings, errors, resource_timeline,
            successful_requests, total_requests, total_duration
        )
        
        return LoadTestResult(
            test_config=test_config,
            performance_metrics=performance_metrics,
            detailed_timings=request_timings,
            error_log=errors,
            resource_timeline=resource_timeline,
            success=performance_metrics.success_rate >= 0.8
        )
    
    async def _simulate_user_load(self, client: TestClient, user_id: int, 
                                 flow_name: str, duration: float, 
                                 requests_per_user: int,
                                 request_timings: List[float], 
                                 errors: List[str]) -> Dict[str, int]:
        """Simulate load from individual user."""
        user_stats = {"successful_requests": 0, "total_requests": 0, "errors": 0}
        
        try:
            # Create session for user
            session_response = client.get("/api/session")
            if session_response.status_code != 200:
                errors.append(f"User {user_id}: Failed to create session")
                return user_stats
            
            session_id = session_response.json()["session_id"]
            
            # Calculate request timing
            request_interval = duration / requests_per_user if requests_per_user > 0 else 1.0
            
            # Simulate user behavior
            for request_num in range(requests_per_user):
                request_start = time.time()
                
                try:
                    # Generate realistic query for flow
                    query = self._generate_user_query(flow_name, user_id, request_num)
                    
                    # Send request via WebSocket (simulate terminal interaction)
                    with client.websocket_connect(f"/ws/{session_id}") as websocket:
                        websocket.send_json({
                            "type": "run_flow",
                            "flow": flow_name,
                            "query": query,
                            "user_id": user_id,
                            "request_id": request_num
                        })
                        
                        # Wait briefly for acknowledgment
                        await asyncio.sleep(0.1)
                    
                    request_duration = time.time() - request_start
                    request_timings.append(request_duration * 1000)  # Convert to ms
                    user_stats["successful_requests"] += 1
                    
                except Exception as e:
                    errors.append(f"User {user_id}, Request {request_num}: {e}")
                    user_stats["errors"] += 1
                
                user_stats["total_requests"] += 1
                
                # Wait before next request
                if request_num < requests_per_user - 1:
                    await asyncio.sleep(request_interval * (0.8 + 0.4 * user_id / concurrent_users))  # Jitter
            
            # Cleanup session
            try:
                client.delete(f"/api/session/{session_id}")
            except:
                pass
                
        except Exception as e:
            errors.append(f"User {user_id} simulation failed: {e}")
        
        return user_stats
    
    def _generate_user_query(self, flow_name: str, user_id: int, request_num: int) -> str:
        """Generate realistic user query for load testing."""
        query_templates = {
            "osb": [
                f"User {user_id} content analysis request {request_num}",
                f"Policy review needed for user content (session {user_id}-{request_num})",
                f"Moderation request from user {user_id}, iteration {request_num}"
            ],
            "content_moderation": [
                f"Classify content from user {user_id}, request {request_num}",
                f"Review submission {request_num} from user {user_id}",
                f"Moderate user content (user: {user_id}, req: {request_num})"
            ],
            "research": [
                f"Research query {request_num} from user {user_id}",
                f"Analyze trends (user {user_id}, request {request_num})",
                f"Study topic submitted by user {user_id}, query {request_num}"
            ]
        }
        
        templates = query_templates.get(flow_name, [f"Generic query {request_num} from user {user_id}"])
        return templates[request_num % len(templates)]
    
    def _calculate_performance_metrics(self, test_config: Dict[str, Any],
                                     request_timings: List[float], errors: List[str],
                                     resource_timeline: List[Dict[str, float]],
                                     successful_requests: int, total_requests: int,
                                     total_duration: float) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        # Latency statistics
        if request_timings:
            avg_latency = statistics.mean(request_timings)
            p95_latency = statistics.quantiles(request_timings, n=20)[18]  # 95th percentile
            p99_latency = statistics.quantiles(request_timings, n=100)[98]  # 99th percentile
        else:
            avg_latency = p95_latency = p99_latency = 0.0
        
        # Resource usage
        if resource_timeline:
            avg_memory = statistics.mean([m["memory_mb"] for m in resource_timeline])
            avg_cpu = statistics.mean([m["cpu_percent"] for m in resource_timeline])
        else:
            avg_memory = avg_cpu = 0.0
        
        # Success rate and throughput
        success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
        throughput = successful_requests / total_duration if total_duration > 0 else 0.0
        
        return PerformanceMetrics(
            test_name=test_config.get("test_name", "load_test"),
            flow_name=test_config["flow_name"],
            duration=total_duration,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            request_count=total_requests,
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_rps=throughput,
            error_count=len(errors),
            concurrent_users=test_config["concurrent_users"]
        )


@pytest.fixture
def performance_app():
    """Create app configured for performance testing."""
    mock_bm = MagicMock(spec=BM)
    mock_flow_runner = MagicMock(spec=FlowRunner)
    
    # Configure multiple flows for testing
    mock_flow_runner.flows = {
        "osb": create_test_flow_config("osb", ["researcher", "policy_analyst", "fact_checker", "explorer"]),
        "content_moderation": create_test_flow_config("content_moderation", ["classifier", "reviewer"]),
        "research": create_test_flow_config("research", ["researcher", "analyst", "synthesizer"])
    }
    
    return create_app(mock_bm, mock_flow_runner)


class TestSystemPerformance:
    """Comprehensive system performance tests."""

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "flow_config",
        [
            {"flow_name": "osb", "concurrent_users": 5, "duration_seconds": 10, "requests_per_user": 3},
            {"flow_name": "content_moderation", "concurrent_users": 8, "duration_seconds": 15, "requests_per_user": 5},
            {"flow_name": "research", "concurrent_users": 3, "duration_seconds": 20, "requests_per_user": 4},
        ],
    )
    async def test_flow_performance_under_load(self, performance_app, flow_config):
        """Test performance of different flows under load."""
        flow_configs = {
            "osb": create_test_flow_config("osb", ["researcher", "policy_analyst", "fact_checker", "explorer"]),
            "content_moderation": create_test_flow_config("content_moderation", ["classifier", "reviewer"]),
            "research": create_test_flow_config("research", ["researcher", "analyst", "synthesizer"])
        }

        load_runner = LoadTestRunner(performance_app, flow_configs)
        result = await load_runner.run_load_test(flow_config)

        # Validate performance requirements
        assert result.success is True, f"Load test failed for {flow_config['flow_name']}"
        assert result.performance_metrics.success_rate >= 0.8, f"Success rate too low: {result.performance_metrics.success_rate}"
        assert result.performance_metrics.avg_latency_ms < 5000, f"Average latency too high: {result.performance_metrics.avg_latency_ms}ms"
        assert result.performance_metrics.p95_latency_ms < 10000, f"P95 latency too high: {result.performance_metrics.p95_latency_ms}ms"

        # Log performance metrics
        print(f"\n{flow_config['flow_name']} Performance Results:")
        print(f"  Success Rate: {result.performance_metrics.success_rate:.1%}")
        print(f"  Avg Latency: {result.performance_metrics.avg_latency_ms:.1f}ms")
        print(f"  P95 Latency: {result.performance_metrics.p95_latency_ms:.1f}ms")
        print(f"  Throughput: {result.performance_metrics.throughput_rps:.1f} req/s")
        print(f"  Memory Usage: {result.performance_metrics.memory_usage_mb:.1f}MB")

    @pytest.mark.anyio
    async def test_scalability_across_flows(self, performance_app):
        """Test system scalability with increasing load across different flows."""
        flow_configs = {
            "osb": create_test_flow_config("osb", ["researcher", "policy_analyst"]),
            "content_moderation": create_test_flow_config("content_moderation", ["classifier", "reviewer"]),
            "research": create_test_flow_config("research", ["researcher", "analyst"])
        }

        load_runner = LoadTestRunner(performance_app, flow_configs)

        # Test different load levels
        load_levels = [
            {"concurrent_users": 2, "duration_seconds": 8, "requests_per_user": 2},
            {"concurrent_users": 5, "duration_seconds": 10, "requests_per_user": 3},
            {"concurrent_users": 8, "duration_seconds": 12, "requests_per_user": 4}
        ]

        scalability_results = {}

        for flow_name in ["osb", "content_moderation", "research"]:
            flow_results = []

            for load_level in load_levels:
                test_config = {**load_level, "flow_name": flow_name, "test_name": f"scalability_{flow_name}"}
                result = await load_runner.run_load_test(test_config)
                flow_results.append(result.performance_metrics)

            scalability_results[flow_name] = flow_results

        # Validate scalability characteristics
        for flow_name, results in scalability_results.items():
            # Performance should not degrade drastically with increased load
            latency_increase = results[-1].avg_latency_ms / results[0].avg_latency_ms
            assert latency_increase < 3.0, f"{flow_name}: Latency increased {latency_increase:.1f}x with load"

            # Success rate should remain high
            min_success_rate = min(r.success_rate for r in results)
            assert min_success_rate >= 0.7, f"{flow_name}: Success rate dropped to {min_success_rate:.1%}"

            print(f"\n{flow_name} Scalability:")
            for i, result in enumerate(results):
                print(f"  Load {i+1}: {result.concurrent_users} users, "
                      f"{result.avg_latency_ms:.1f}ms avg, {result.success_rate:.1%} success")

    @pytest.mark.anyio
    async def test_memory_usage_stability(self, performance_app):
        """Test memory usage stability over extended periods."""
        flow_configs = {"osb": create_test_flow_config("osb", ["researcher", "policy_analyst"])}
        load_runner = LoadTestRunner(performance_app, flow_configs)

        # Extended test with moderate load
        test_config = {
            "flow_name": "osb",
            "concurrent_users": 3,
            "duration_seconds": 30,  # Longer duration
            "requests_per_user": 8,
            "test_name": "memory_stability"
        }

        result = await load_runner.run_load_test(test_config)

        # Analyze memory usage over time
        memory_timeline = [m["memory_mb"] for m in result.resource_timeline]

        if len(memory_timeline) > 5:
            # Check for memory leaks (significant upward trend)
            start_memory = statistics.mean(memory_timeline[:3])
            end_memory = statistics.mean(memory_timeline[-3:])
            memory_growth = (end_memory - start_memory) / start_memory

            assert memory_growth < 0.5, f"Potential memory leak: {memory_growth:.1%} growth"

            # Check memory stability (low variance)
            memory_std = statistics.stdev(memory_timeline)
            memory_cv = memory_std / statistics.mean(memory_timeline)  # Coefficient of variation

            assert memory_cv < 0.3, f"Memory usage too unstable: CV={memory_cv:.2f}"

            print(f"\nMemory Stability Results:")
            print(f"  Start Memory: {start_memory:.1f}MB")
            print(f"  End Memory: {end_memory:.1f}MB")
            print(f"  Growth: {memory_growth:.1%}")
            print(f"  Stability (CV): {memory_cv:.2f}")

    @pytest.mark.anyio
    async def test_concurrent_flow_performance(self, performance_app):
        """Test performance when multiple flows run concurrently."""
        flow_configs = {
            "osb": create_test_flow_config("osb", ["researcher", "policy_analyst"]),
            "content_moderation": create_test_flow_config("content_moderation", ["classifier"]),
            "research": create_test_flow_config("research", ["researcher"])
        }

        load_runner = LoadTestRunner(performance_app, flow_configs)

        # Run all flows concurrently
        concurrent_tests = [
            {"flow_name": "osb", "concurrent_users": 3, "duration_seconds": 15, "requests_per_user": 3},
            {"flow_name": "content_moderation", "concurrent_users": 4, "duration_seconds": 15, "requests_per_user": 4},
            {"flow_name": "research", "concurrent_users": 2, "duration_seconds": 15, "requests_per_user": 3}
        ]

        # Execute all tests concurrently
        test_tasks = [
            asyncio.create_task(load_runner.run_load_test(test_config))
            for test_config in concurrent_tests
        ]

        results = await asyncio.gather(*test_tasks, return_exceptions=True)

        # Validate all flows performed acceptably under concurrent load
        successful_flows = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Flow {concurrent_tests[i]['flow_name']} failed: {result}")
            else:
                assert result.success is True
                assert result.performance_metrics.success_rate >= 0.7
                successful_flows += 1

                print(f"\n{result.performance_metrics.flow_name} (Concurrent):")
                print(f"  Success Rate: {result.performance_metrics.success_rate:.1%}")
                print(f"  Avg Latency: {result.performance_metrics.avg_latency_ms:.1f}ms")

        # At least 2 out of 3 flows should succeed under concurrent load
        assert successful_flows >= 2, f"Only {successful_flows}/3 flows succeeded under concurrent load"


class TestReliabilityAndStress:
    """Reliability and stress testing for edge conditions."""

    @pytest.mark.anyio
    async def test_error_recovery_under_load(self, performance_app):
        """Test system recovery when errors occur under load."""
        flow_configs = {"osb": create_test_flow_config("osb", ["researcher"])}
        load_runner = LoadTestRunner(performance_app, flow_configs)

        # Test with configuration that may cause some errors
        test_config = {
            "flow_name": "osb",
            "concurrent_users": 6,  # Higher load
            "duration_seconds": 10,
            "requests_per_user": 5,
            "test_name": "error_recovery"
        }

        result = await load_runner.run_load_test(test_config)

        # System should handle errors gracefully
        error_rate = result.performance_metrics.error_count / result.performance_metrics.request_count
        assert error_rate < 0.3, f"Error rate too high: {error_rate:.1%}"

        # Even with errors, some requests should succeed
        assert result.performance_metrics.success_rate >= 0.5, f"Success rate too low: {result.performance_metrics.success_rate:.1%}"

        print(f"\nError Recovery Results:")
        print(f"  Error Rate: {error_rate:.1%}")
        print(f"  Success Rate: {result.performance_metrics.success_rate:.1%}")
        print(f"  Total Errors: {result.performance_metrics.error_count}")

    @pytest.mark.anyio
    async def test_burst_load_handling(self, performance_app):
        """Test system handling of sudden burst loads."""
        flow_configs = {"content_moderation": create_test_flow_config("content_moderation", ["classifier"])}
        load_runner = LoadTestRunner(performance_app, flow_configs)

        # Simulate burst load - many users, short duration, many requests
        burst_config = {
            "flow_name": "content_moderation",
            "concurrent_users": 10,
            "duration_seconds": 5,  # Short burst
            "requests_per_user": 3,
            "test_name": "burst_load"
        }

        result = await load_runner.run_load_test(burst_config)

        # System should handle burst without complete failure
        assert result.performance_metrics.success_rate >= 0.6, f"Burst handling failed: {result.performance_metrics.success_rate:.1%}"

        # Latency may be higher but should be reasonable
        assert result.performance_metrics.avg_latency_ms < 8000, f"Burst latency too high: {result.performance_metrics.avg_latency_ms:.1f}ms"

        print(f"\nBurst Load Results:")
        print(f"  Success Rate: {result.performance_metrics.success_rate:.1%}")
        print(f"  Peak Latency: {result.performance_metrics.p99_latency_ms:.1f}ms")
        print(f"  Throughput: {result.performance_metrics.throughput_rps:.1f} req/s")

    @pytest.mark.anyio
    async def test_resource_exhaustion_protection(self, performance_app):
        """Test system protection against resource exhaustion."""
        flow_configs = {"research": create_test_flow_config("research", ["researcher", "analyst"])}
        load_runner = LoadTestRunner(performance_app, flow_configs)

        # Test with very high load to approach resource limits
        exhaustion_config = {
            "flow_name": "research",
            "concurrent_users": 15,  # Very high load
            "duration_seconds": 8,
            "requests_per_user": 2,
            "test_name": "resource_exhaustion"
        }

        result = await load_runner.run_load_test(exhaustion_config)

        # System should either succeed or fail gracefully (not crash)
        # Resource usage should be bounded
        max_memory = max([m["memory_mb"] for m in result.resource_timeline]) if result.resource_timeline else 0
        assert max_memory < 1000, f"Memory usage too high: {max_memory:.1f}MB"  # Reasonable limit

        # System should maintain some level of service
        assert result.performance_metrics.success_rate >= 0.3, f"System completely failed under load: {result.performance_metrics.success_rate:.1%}"

        print(f"\nResource Exhaustion Protection:")
        print(f"  Success Rate: {result.performance_metrics.success_rate:.1%}")
        print(f"  Max Memory: {max_memory:.1f}MB")
        print(f"  Error Count: {result.performance_metrics.error_count}")


@pytest.mark.slow
class TestLongRunningPerformance:
    """Long-running performance tests for production validation."""

    @pytest.mark.anyio
    async def test_sustained_production_load(self, performance_app):
        """Test sustained load similar to production conditions."""
        flow_configs = {
            "osb": create_test_flow_config("osb", ["researcher", "policy_analyst"]),
            "content_moderation": create_test_flow_config("content_moderation", ["classifier", "reviewer"])
        }

        load_runner = LoadTestRunner(performance_app, flow_configs)

        # Simulate realistic production load
        production_config = {
            "flow_name": "osb",
            "concurrent_users": 4,
            "duration_seconds": 60,  # 1 minute sustained
            "requests_per_user": 10,
            "test_name": "production_simulation"
        }

        result = await load_runner.run_load_test(production_config)

        # Production requirements
        assert result.success is True, "Production simulation failed"
        assert result.performance_metrics.success_rate >= 0.95, f"Production success rate too low: {result.performance_metrics.success_rate:.1%}"
        assert result.performance_metrics.avg_latency_ms < 3000, f"Production latency too high: {result.performance_metrics.avg_latency_ms:.1f}ms"
        assert result.performance_metrics.p95_latency_ms < 5000, f"Production P95 latency too high: {result.performance_metrics.p95_latency_ms:.1f}ms"

        print(f"\nProduction Simulation Results:")
        print(f"  Duration: {result.performance_metrics.duration:.1f}s")
        print(f"  Total Requests: {result.performance_metrics.request_count}")
        print(f"  Success Rate: {result.performance_metrics.success_rate:.1%}")
        print(f"  Avg Latency: {result.performance_metrics.avg_latency_ms:.1f}ms")
        print(f"  P95 Latency: {result.performance_metrics.p95_latency_ms:.1f}ms")
        print(f"  Throughput: {result.performance_metrics.throughput_rps:.1f} req/s")
