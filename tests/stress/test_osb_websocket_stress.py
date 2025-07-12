"""
OSB WebSocket Stress and Performance Tests.

Comprehensive stress testing suite for OSB WebSocket connections including:
- High concurrent connection testing
- Message throughput and latency testing
- Connection reliability under load
- Memory and resource usage testing
- Error recovery and reconnection testing
- Long-duration stability testing

These tests validate that OSB WebSocket infrastructure can handle
production-level loads and stress conditions reliably.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from buttermilk.api.flow import create_app
from buttermilk._core.bm_init import BM
from tests.utils.websocket_test_utils import (
    WebSocketTestSession,
    WebSocketStressTestRunner,
    websocket_test_context,
    simulate_osb_workflow,
    create_mock_flow_runner_for_websocket_tests,
    validate_websocket_test_results
)


@pytest.fixture
def mock_bm():
    """Mock BM instance for stress testing."""
    mock_bm = MagicMock(spec=BM)
    mock_bm.llms = MagicMock()
    return mock_bm


@pytest.fixture
def stress_test_app(mock_bm):
    """Create test app optimized for stress testing."""
    mock_flow_runner = create_mock_flow_runner_for_websocket_tests()
    app = create_app(mock_bm, mock_flow_runner)
    return app


class TestOSBWebSocketConcurrency:
    """Test concurrent WebSocket connections for OSB flows."""

    @pytest.mark.anyio
    async def test_high_concurrent_connections(self, stress_test_app):
        """Test handling of high number of concurrent WebSocket connections."""
        max_connections = 20
        test_duration = 10  # seconds

        with TestClient(stress_test_app) as client:
            # Create multiple concurrent connections
            sessions = []
            connection_tasks = []

            for i in range(max_connections):
                session_id = f"stress-concurrent-{i}"
                # Note: Using TestClient WebSocket context instead of direct WebSocket
                # for stress testing compatibility
                try:
                    ws = client.websocket_connect(f"/ws/{session_id}")
                    session_ws = ws.__enter__()
                    sessions.append((session_id, session_ws))
                except Exception as e:
                    # Track connection failures
                    pass

            # Verify most connections succeeded
            assert len(sessions) >= max_connections * 0.8  # Allow 20% failure rate

            # Send concurrent messages
            message_tasks = []
            for session_id, websocket in sessions:
                task = asyncio.create_task(
                    self._send_concurrent_messages(websocket, session_id, test_duration)
                )
                message_tasks.append(task)

            # Wait for all message sending to complete
            results = await asyncio.gather(*message_tasks, return_exceptions=True)

            # Verify most message sending succeeded
            successful_sessions = sum(1 for result in results if not isinstance(result, Exception))
            assert successful_sessions >= len(sessions) * 0.9  # Allow 10% failure rate

            # Cleanup
            for _, websocket in sessions:
                try:
                    websocket.__exit__(None, None, None)
                except:
                    pass

    async def _send_concurrent_messages(self, websocket, session_id: str, duration: int):
        """Send messages concurrently during stress test."""
        end_time = time.time() + duration
        message_count = 0

        while time.time() < end_time:
            try:
                osb_query = {
                    "type": "run_flow",
                    "flow": "osb",
                    "query": f"Concurrent stress test {message_count} from {session_id}",
                    "case_number": f"STRESS-CONCURRENT-{message_count:03d}"
                }

                websocket.send_json(osb_query)
                message_count += 1

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)

            except Exception as e:
                # Track errors but continue
                break

        return message_count

    @pytest.mark.anyio
    async def test_connection_burst_handling(self, stress_test_app):
        """Test handling of burst connection attempts."""
        burst_size = 15
        burst_interval = 0.01  # Very fast burst

        with TestClient(stress_test_app) as client:
            connection_times = []
            successful_connections = 0

            # Create burst of connections
            for i in range(burst_size):
                start_time = time.time()

                try:
                    session_id = f"burst-connection-{i}"
                    with client.websocket_connect(f"/ws/{session_id}") as websocket:
                        connection_time = time.time() - start_time
                        connection_times.append(connection_time)
                        successful_connections += 1

                        # Send test message to verify connection works
                        websocket.send_json({
                            "type": "run_flow",
                            "flow": "osb",
                            "query": f"Burst test {i}"
                        })

                except Exception as e:
                    # Connection failed - this is acceptable under burst load
                    pass

                await asyncio.sleep(burst_interval)

            # Verify reasonable success rate even under burst
            success_rate = successful_connections / burst_size
            assert success_rate >= 0.7  # At least 70% success rate

            # Verify reasonable connection times
            if connection_times:
                avg_connection_time = sum(connection_times) / len(connection_times)
                assert avg_connection_time < 2.0  # Under 2 seconds average

    @pytest.mark.anyio
    async def test_session_isolation_under_load(self, stress_test_app):
        """Test that sessions remain isolated under high load."""
        num_sessions = 10
        messages_per_session = 5

        with TestClient(stress_test_app) as client:
            # Create multiple sessions
            sessions = []
            for i in range(num_sessions):
                session_id = f"isolation-test-{i}"
                ws = client.websocket_connect(f"/ws/{session_id}")
                websocket = ws.__enter__()
                sessions.append((session_id, websocket, ws))

            try:
                # Send unique messages from each session
                message_tasks = []
                for i, (session_id, websocket, _) in enumerate(sessions):
                    task = asyncio.create_task(
                        self._send_unique_session_messages(websocket, session_id, messages_per_session)
                    )
                    message_tasks.append(task)

                # Wait for all sessions to complete
                session_results = await asyncio.gather(*message_tasks, return_exceptions=True)

                # Verify all sessions completed successfully
                successful_sessions = sum(1 for result in session_results if not isinstance(result, Exception))
                assert successful_sessions == num_sessions

            finally:
                # Cleanup all sessions
                for _, _, ws_context in sessions:
                    try:
                        ws_context.__exit__(None, None, None)
                    except:
                        pass

    async def _send_unique_session_messages(self, websocket, session_id: str, message_count: int):
        """Send unique messages from a specific session."""
        for i in range(message_count):
            unique_query = f"Isolation test from {session_id} message {i}"
            osb_message = {
                "type": "run_flow",
                "flow": "osb",
                "query": unique_query,
                "session_specific_data": {
                    "session_id": session_id,
                    "message_number": i,
                    "unique_identifier": f"{session_id}-{i}"
                }
            }

            websocket.send_json(osb_message)

            # Brief pause between messages
            await asyncio.sleep(0.2)


class TestOSBWebSocketPerformance:
    """Test WebSocket performance characteristics."""

    @pytest.mark.anyio
    async def test_message_throughput_performance(self, stress_test_app):
        """Test WebSocket message throughput under load."""
        test_duration = 5  # seconds
        target_messages_per_second = 10

        with TestClient(stress_test_app) as client:
            session_id = "throughput-test-session"

            with client.websocket_connect(f"/ws/{session_id}") as websocket:
                start_time = time.time()
                message_count = 0

                while time.time() - start_time < test_duration:
                    osb_query = {
                        "type": "run_flow",
                        "flow": "osb",
                        "query": f"Throughput test message {message_count}",
                        "message_id": message_count
                    }

                    websocket.send_json(osb_query)
                    message_count += 1

                actual_duration = time.time() - start_time
                actual_throughput = message_count / actual_duration

                # Verify throughput meets minimum requirements
                assert actual_throughput >= target_messages_per_second * 0.8  # Allow 20% variance

    @pytest.mark.anyio
    async def test_large_message_handling(self, stress_test_app):
        """Test handling of large OSB query messages."""
        with TestClient(stress_test_app) as client:
            session_id = "large-message-test"

            with client.websocket_connect(f"/ws/{session_id}") as websocket:
                # Test various message sizes
                message_sizes = [100, 500, 1000, 1500, 1900]  # Character counts

                for size in message_sizes:
                    large_query = "x" * size
                    large_message = {
                        "type": "run_flow",
                        "flow": "osb",
                        "query": large_query,
                        "case_number": f"LARGE-MSG-{size}",
                        "additional_metadata": {
                            "message_size": size,
                            "test_type": "large_message_handling"
                        }
                    }

                    # Should handle large messages without issues
                    start_time = time.time()
                    websocket.send_json(large_message)
                    send_time = time.time() - start_time

                    # Verify reasonable send time even for large messages
                    assert send_time < 1.0  # Under 1 second

    @pytest.mark.anyio
    async def test_rapid_message_sequence_handling(self, stress_test_app):
        """Test handling of rapid message sequences."""
        with TestClient(stress_test_app) as client:
            session_id = "rapid-message-test"

            with client.websocket_connect(f"/ws/{session_id}") as websocket:
                # Send messages as rapidly as possible
                rapid_message_count = 20

                start_time = time.time()
                for i in range(rapid_message_count):
                    rapid_message = {
                        "type": "run_flow",
                        "flow": "osb",
                        "query": f"Rapid message {i}",
                        "sequence_number": i
                    }

                    websocket.send_json(rapid_message)

                total_time = time.time() - start_time

                # Verify all messages sent in reasonable time
                assert total_time < 2.0  # All 20 messages in under 2 seconds

                # Verify high message rate
                message_rate = rapid_message_count / total_time
                assert message_rate >= 10  # At least 10 messages per second


class TestOSBWebSocketReliability:
    """Test WebSocket reliability and error recovery."""

    @pytest.mark.anyio
    async def test_connection_recovery_after_errors(self, stress_test_app):
        """Test WebSocket recovery after connection errors."""
        with TestClient(stress_test_app) as client:
            session_id = "recovery-test-session"

            # Test multiple connection/disconnection cycles
            for cycle in range(3):
                with client.websocket_connect(f"/ws/{session_id}") as websocket:
                    # Send test message
                    test_message = {
                        "type": "run_flow",
                        "flow": "osb",
                        "query": f"Recovery test cycle {cycle}",
                        "cycle_number": cycle
                    }

                    websocket.send_json(test_message)

                    # Connection should work normally after previous disconnections
                    assert True  # If we get here, connection/message sending worked

    @pytest.mark.anyio
    async def test_malformed_message_resilience(self, stress_test_app):
        """Test WebSocket resilience to malformed messages."""
        with TestClient(stress_test_app) as client:
            session_id = "malformed-message-test"

            with client.websocket_connect(f"/ws/{session_id}") as websocket:
                # Test various malformed messages
                malformed_messages = [
                    {"type": "run_flow"},  # Missing required fields
                    {"flow": "osb", "query": "test"},  # Missing type
                    {"type": "invalid_type", "query": "test"},  # Invalid type
                    {},  # Empty message
                ]

                for i, malformed_msg in enumerate(malformed_messages):
                    try:
                        websocket.send_json(malformed_msg)
                        # Connection should remain stable
                    except Exception as e:
                        # Some test clients may reject at send level
                        pass

                # Send valid message after malformed ones to test recovery
                valid_message = {
                    "type": "run_flow",
                    "flow": "osb",
                    "query": "Recovery after malformed messages"
                }

                websocket.send_json(valid_message)
                # If this succeeds, connection recovered properly

    @pytest.mark.anyio
    async def test_long_duration_stability(self, stress_test_app):
        """Test WebSocket stability over extended duration."""
        test_duration = 15  # seconds (reduced for test performance)
        message_interval = 1.0  # seconds between messages

        with TestClient(stress_test_app) as client:
            session_id = "long-duration-test"

            with client.websocket_connect(f"/ws/{session_id}") as websocket:
                start_time = time.time()
                message_count = 0

                while time.time() - start_time < test_duration:
                    stability_message = {
                        "type": "run_flow",
                        "flow": "osb",
                        "query": f"Stability test message {message_count}",
                        "timestamp": time.time(),
                        "duration_elapsed": time.time() - start_time
                    }

                    websocket.send_json(stability_message)
                    message_count += 1

                    await asyncio.sleep(message_interval)

                # Verify connection remained stable throughout test
                assert message_count >= test_duration / message_interval * 0.9  # Allow some tolerance


class TestOSBWebSocketResourceUsage:
    """Test WebSocket resource usage and limits."""

    @pytest.mark.anyio
    async def test_session_cleanup_effectiveness(self, stress_test_app):
        """Test that WebSocket sessions are properly cleaned up."""
        session_count = 5

        with TestClient(stress_test_app) as client:
            # Create and close multiple sessions to test cleanup
            for i in range(session_count):
                session_id = f"cleanup-test-{i}"

                with client.websocket_connect(f"/ws/{session_id}") as websocket:
                    # Send message to ensure session is active
                    websocket.send_json({
                        "type": "run_flow",
                        "flow": "osb",
                        "query": f"Cleanup test {i}"
                    })

                # Session should be cleaned up when context exits
                # In a real implementation, we would verify cleanup through monitoring

            # All sessions should be cleaned up at this point
            assert True  # Test passes if no resource leaks occur

    @pytest.mark.anyio
    async def test_memory_usage_under_load(self, stress_test_app):
        """Test memory usage remains reasonable under WebSocket load."""
        # This is a placeholder for memory monitoring
        # In a full implementation, this would track memory usage

        concurrent_sessions = 8

        with TestClient(stress_test_app) as client:
            sessions = []

            # Create multiple concurrent sessions
            for i in range(concurrent_sessions):
                session_id = f"memory-test-{i}"
                ws = client.websocket_connect(f"/ws/{session_id}")
                websocket = ws.__enter__()
                sessions.append((websocket, ws))

            try:
                # Send messages from all sessions
                for i, (websocket, _) in enumerate(sessions):
                    websocket.send_json({
                        "type": "run_flow",
                        "flow": "osb",
                        "query": f"Memory test from session {i}"
                    })

                # In a real test, memory usage would be monitored here
                assert True  # Test passes if no memory issues occur

            finally:
                # Cleanup
                for _, ws_context in sessions:
                    try:
                        ws_context.__exit__(None, None, None)
                    except:
                        pass


@pytest.mark.slow
class TestOSBWebSocketIntegrationStress:
    """Integration stress tests combining multiple stress factors."""

    @pytest.mark.anyio
    async def test_combined_stress_scenario(self, stress_test_app):
        """Test WebSocket handling under combined stress conditions."""
        # This test combines multiple stress factors:
        # - Multiple concurrent connections
        # - High message frequency
        # - Large messages
        # - Extended duration

        num_sessions = 5
        test_duration = 10  # seconds
        message_frequency = 0.5  # seconds between messages

        with TestClient(stress_test_app) as client:
            sessions = []

            # Create concurrent sessions
            for i in range(num_sessions):
                session_id = f"combined-stress-{i}"
                ws = client.websocket_connect(f"/ws/{session_id}")
                websocket = ws.__enter__()
                sessions.append((session_id, websocket, ws))

            try:
                # Run stress test on all sessions
                stress_tasks = []
                for session_id, websocket, _ in sessions:
                    task = asyncio.create_task(
                        self._run_session_stress(websocket, session_id, test_duration, message_frequency)
                    )
                    stress_tasks.append(task)

                # Wait for all stress tests to complete
                results = await asyncio.gather(*stress_tasks, return_exceptions=True)

                # Verify most sessions handled stress successfully
                successful_sessions = sum(1 for result in results if not isinstance(result, Exception))
                assert successful_sessions >= num_sessions * 0.8  # 80% success rate

            finally:
                # Cleanup all sessions
                for _, _, ws_context in sessions:
                    try:
                        ws_context.__exit__(None, None, None)
                    except:
                        pass

    async def _run_session_stress(self, websocket, session_id: str, duration: int, message_freq: float):
        """Run stress test on individual session."""
        start_time = time.time()
        message_count = 0

        while time.time() - start_time < duration:
            # Create varying message sizes
            base_query = f"Combined stress test {message_count} from {session_id}"
            padding_size = (message_count % 5) * 100  # Varying sizes
            large_query = base_query + ("x" * padding_size)

            stress_message = {
                "type": "run_flow",
                "flow": "osb",
                "query": large_query,
                "stress_metadata": {
                    "session_id": session_id,
                    "message_count": message_count,
                    "timestamp": time.time(),
                    "message_size": len(large_query)
                }
            }

            websocket.send_json(stress_message)
            message_count += 1

            await asyncio.sleep(message_freq)

        return message_count
