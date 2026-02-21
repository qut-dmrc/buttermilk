"""
WebSocket Testing Utilities for OSB Flow Testing.

This module provides comprehensive utilities for testing WebSocket functionality
in OSB (Oversight Board) interactive flows, including:

- WebSocket connection management and lifecycle testing
- Message validation and flow testing
- Performance and stress testing utilities
- Concurrent session management testing
- Error simulation and recovery testing

Designed to support both unit tests and integration tests for WebSocket-based
OSB functionality with comprehensive validation and monitoring capabilities.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import websockets

from buttermilk import logger


class WebSocketTestSession:
    """Enhanced WebSocket test session with comprehensive monitoring."""

    def __init__(self, session_id: str, uri: str):
        self.session_id = session_id
        self.uri = uri
        self.websocket = None
        self.is_connected = False

        # Message tracking
        self.messages_sent = []
        self.messages_received = []
        self.connection_events = []

        # Performance metrics
        self.connection_time = None
        self.message_latencies = []
        self.error_count = 0

        # OSB-specific tracking
        self.osb_queries_sent = 0
        self.osb_responses_received = 0
        self.agent_interactions = {}

    async def connect(self, timeout: float = 10.0) -> bool:
        """Establish WebSocket connection with timeout."""
        start_time = time.time()

        try:
<<<<<<< HEAD
            self.websocket = await asyncio.wait_for(websockets.connect(self.uri), timeout=timeout)
=======
            self.websocket = await asyncio.wait_for(
                websockets.connect(self.uri), timeout=timeout
            )
>>>>>>> origin/stable

            self.connection_time = time.time() - start_time
            self.is_connected = True
            self.connection_events.append(("connected", time.time()))

            logger.debug(
                "WebSocket connected",
                session_id=self.session_id,
                connection_time=f"{self.connection_time:.3f}s",
            )
            return True

        except asyncio.TimeoutError:
            self.connection_events.append(("connect_timeout", time.time()))
            logger.warning("WebSocket connection timeout", session_id=self.session_id)
            return False
        except Exception as e:
            self.connection_events.append(("connect_error", time.time(), str(e)))
<<<<<<< HEAD
            logger.error("WebSocket connection failed", session_id=self.session_id, error=e)
=======
            logger.error(
                "WebSocket connection failed", session_id=self.session_id, error=e
            )
>>>>>>> origin/stable
            return False

    async def disconnect(self):
        """Close WebSocket connection gracefully."""
        if self.websocket and self.is_connected:
            try:
                await self.websocket.close()
                self.is_connected = False
                self.connection_events.append(("disconnected", time.time()))
                logger.debug("WebSocket disconnected", session_id=self.session_id)
            except Exception as e:
                logger.warning(
                    "Error during WebSocket disconnect",
                    session_id=self.session_id,
                    error=e,
                )

<<<<<<< HEAD
    async def send_message(self, message: Dict[str, Any], track_latency: bool = True) -> bool:
=======
    async def send_message(
        self, message: Dict[str, Any], track_latency: bool = True
    ) -> bool:
>>>>>>> origin/stable
        """Send message with optional latency tracking."""
        if not self.is_connected or not self.websocket:
            return False

        try:
            send_time = time.time()
            message_json = json.dumps(message)

            await self.websocket.send(message_json)

            # Track message
<<<<<<< HEAD
            self.messages_sent.append({"message": message, "timestamp": send_time, "size": len(message_json)})
=======
            self.messages_sent.append(
                {"message": message, "timestamp": send_time, "size": len(message_json)}
            )
>>>>>>> origin/stable

            # Track OSB-specific metrics
            if message.get("type") == "run_flow" and message.get("flow") == "osb":
                self.osb_queries_sent += 1

<<<<<<< HEAD
            self.connection_events.append(("message_sent", send_time, message.get("type", "unknown")))
=======
            self.connection_events.append(
                ("message_sent", send_time, message.get("type", "unknown"))
            )
>>>>>>> origin/stable
            return True

        except Exception as e:
            self.error_count += 1
            self.connection_events.append(("send_error", time.time(), str(e)))
<<<<<<< HEAD
            logger.error("Failed to send message in session", session_id=self.session_id, error=e)
=======
            logger.error(
                "Failed to send message in session", session_id=self.session_id, error=e
            )
>>>>>>> origin/stable
            return False

    async def receive_message(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Receive message with timeout."""
        if not self.is_connected or not self.websocket:
            return None

        try:
            receive_time = time.time()
            message_str = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)

            message = json.loads(message_str)

            # Track message
            self.messages_received.append(
                {
                    "message": message,
                    "timestamp": receive_time,
                    "size": len(message_str),
                }
            )

            # Track OSB-specific metrics
            if message.get("type") in ["osb_status", "osb_partial", "osb_complete"]:
                self.osb_responses_received += 1

                # Track agent interactions
                if "agent" in message:
                    agent = message["agent"]
                    if agent not in self.agent_interactions:
                        self.agent_interactions[agent] = 0
                    self.agent_interactions[agent] += 1

<<<<<<< HEAD
            self.connection_events.append(("message_received", receive_time, message.get("type", "unknown")))
=======
            self.connection_events.append(
                ("message_received", receive_time, message.get("type", "unknown"))
            )
>>>>>>> origin/stable
            return message

        except asyncio.TimeoutError:
            self.connection_events.append(("receive_timeout", time.time()))
            return None
        except Exception as e:
            self.error_count += 1
            self.connection_events.append(("receive_error", time.time(), str(e)))
            logger.error(
                "Failed to receive message in session",
                session_id=self.session_id,
                error=e,
            )
            return None

    async def send_osb_query(self, query: str, **kwargs) -> bool:
        """Send OSB-specific query message."""
        osb_message = {
            "type": "run_flow",
            "flow": "osb",
            "query": query,
            "session_id": self.session_id,
            **kwargs,
        }
        return await self.send_message(osb_message)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for this session."""
        return {
            "session_id": self.session_id,
            "connection_time": self.connection_time,
            "total_messages_sent": len(self.messages_sent),
            "total_messages_received": len(self.messages_received),
            "error_count": self.error_count,
            "osb_queries_sent": self.osb_queries_sent,
            "osb_responses_received": self.osb_responses_received,
            "agent_interactions": self.agent_interactions,
            "message_latencies": self.message_latencies,
            "average_message_size": self._calculate_average_message_size(),
            "connection_events_count": len(self.connection_events),
        }

    def _calculate_average_message_size(self) -> float:
        """Calculate average message size across all messages."""
        all_messages = self.messages_sent + self.messages_received
        if not all_messages:
            return 0.0

        total_size = sum(msg["size"] for msg in all_messages)
        return total_size / len(all_messages)


class WebSocketStressTestRunner:
    """Stress testing runner for WebSocket connections."""

    def __init__(self, base_uri: str, max_concurrent_sessions: int = 10):
        self.base_uri = base_uri
        self.max_concurrent_sessions = max_concurrent_sessions
        self.sessions: List[WebSocketTestSession] = []
        self.results = {}

<<<<<<< HEAD
    async def run_concurrent_connection_test(self, num_sessions: int = 5, duration_seconds: int = 30) -> Dict[str, Any]:
=======
    async def run_concurrent_connection_test(
        self, num_sessions: int = 5, duration_seconds: int = 30
    ) -> Dict[str, Any]:
>>>>>>> origin/stable
        """Test concurrent WebSocket connections under load."""
        logger.info(
            "Starting concurrent connection test",
            num_sessions=num_sessions,
            duration_seconds=duration_seconds,
        )

        start_time = time.time()

        # Create and connect sessions
        connect_tasks = []
        for i in range(num_sessions):
            session_id = f"stress-test-session-{i}"
<<<<<<< HEAD
            session = WebSocketTestSession(session_id, f"{self.base_uri}/ws/{session_id}")
=======
            session = WebSocketTestSession(
                session_id, f"{self.base_uri}/ws/{session_id}"
            )
>>>>>>> origin/stable
            self.sessions.append(session)
            connect_tasks.append(session.connect())

        # Wait for all connections
<<<<<<< HEAD
        connection_results = await asyncio.gather(*connect_tasks, return_exceptions=True)
        successful_connections = sum(1 for result in connection_results if result is True)
=======
        connection_results = await asyncio.gather(
            *connect_tasks, return_exceptions=True
        )
        successful_connections = sum(
            1 for result in connection_results if result is True
        )
>>>>>>> origin/stable

        logger.info(
            "Connected sessions",
            successful_connections=successful_connections,
            num_sessions=num_sessions,
        )

        # Run message exchange test
        message_tasks = []
        for session in self.sessions:
            if session.is_connected:
<<<<<<< HEAD
                message_tasks.append(self._session_message_loop(session, duration_seconds))
=======
                message_tasks.append(
                    self._session_message_loop(session, duration_seconds)
                )
>>>>>>> origin/stable

        # Wait for message exchange to complete
        await asyncio.gather(*message_tasks, return_exceptions=True)

        # Disconnect all sessions
        disconnect_tasks = [session.disconnect() for session in self.sessions]
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        total_time = time.time() - start_time

        # Compile results
        results = {
            "test_duration": total_time,
            "sessions_requested": num_sessions,
            "sessions_connected": successful_connections,
            "connection_success_rate": successful_connections / num_sessions,
<<<<<<< HEAD
            "session_metrics": [session.get_performance_metrics() for session in self.sessions],
=======
            "session_metrics": [
                session.get_performance_metrics() for session in self.sessions
            ],
>>>>>>> origin/stable
            "aggregate_metrics": self._calculate_aggregate_metrics(),
        }

        self.results = results
        return results

<<<<<<< HEAD
    async def _session_message_loop(self, session: WebSocketTestSession, duration_seconds: int):
=======
    async def _session_message_loop(
        self, session: WebSocketTestSession, duration_seconds: int
    ):
>>>>>>> origin/stable
        """Message exchange loop for individual session during stress test."""
        end_time = time.time() + duration_seconds
        message_count = 0

        while time.time() < end_time and session.is_connected:
            try:
                # Send OSB query
<<<<<<< HEAD
                query = f"Stress test query {message_count + 1} from {session.session_id}"
                await session.send_osb_query(query, case_number=f"STRESS-{message_count:03d}")
=======
                query = (
                    f"Stress test query {message_count + 1} from {session.session_id}"
                )
                await session.send_osb_query(
                    query, case_number=f"STRESS-{message_count:03d}"
                )
>>>>>>> origin/stable

                # Brief pause between messages
                await asyncio.sleep(0.5)
                message_count += 1

            except Exception as e:
<<<<<<< HEAD
                logger.error("Error in message loop", session_id=session.session_id, error=e)
=======
                logger.error(
                    "Error in message loop", session_id=session.session_id, error=e
                )
>>>>>>> origin/stable
                break

    def _calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics across all test sessions."""
        if not self.sessions:
            return {}

        total_messages_sent = sum(len(s.messages_sent) for s in self.sessions)
        total_messages_received = sum(len(s.messages_received) for s in self.sessions)
        total_errors = sum(s.error_count for s in self.sessions)

<<<<<<< HEAD
        connection_times = [s.connection_time for s in self.sessions if s.connection_time]
=======
        connection_times = [
            s.connection_time for s in self.sessions if s.connection_time
        ]
>>>>>>> origin/stable

        return {
            "total_messages_sent": total_messages_sent,
            "total_messages_received": total_messages_received,
            "total_errors": total_errors,
<<<<<<< HEAD
            "average_connection_time": sum(connection_times) / len(connection_times) if connection_times else 0,
=======
            "average_connection_time": sum(connection_times) / len(connection_times)
            if connection_times
            else 0,
>>>>>>> origin/stable
            "max_connection_time": max(connection_times) if connection_times else 0,
            "min_connection_time": min(connection_times) if connection_times else 0,
            "error_rate": total_errors / (total_messages_sent + total_messages_received)
            if (total_messages_sent + total_messages_received) > 0
            else 0,
        }


class WebSocketMessageValidator:
    """Validator for OSB WebSocket message formats and flows."""

    @staticmethod
    def validate_osb_message(message: Dict[str, Any]) -> tuple[bool, str]:
        """Validate OSB message format."""
        if not isinstance(message, dict):
            return False, "Message must be a dictionary"

        message_type = message.get("type")
        if not message_type:
            return False, "Message must have a 'type' field"

        # Validate based on message type
        if message_type == "run_flow":
            return WebSocketMessageValidator._validate_run_flow_message(message)
        elif message_type == "osb_status":
            return WebSocketMessageValidator._validate_osb_status_message(message)
        elif message_type == "osb_partial":
            return WebSocketMessageValidator._validate_osb_partial_message(message)
        elif message_type == "osb_complete":
            return WebSocketMessageValidator._validate_osb_complete_message(message)
        elif message_type == "osb_error":
            return WebSocketMessageValidator._validate_osb_error_message(message)
        else:
            return True, ""  # Allow unknown message types for extensibility

    @staticmethod
    def _validate_run_flow_message(message: Dict[str, Any]) -> tuple[bool, str]:
        """Validate run_flow message format."""
        required_fields = ["flow", "query"]

        for field in required_fields:
            if field not in message:
                return False, f"Missing required field: {field}"

        if message.get("flow") != "osb":
            return False, "Flow must be 'osb' for OSB messages"

        query = message.get("query", "")
        if not query.strip():
            return False, "Query cannot be empty"

        if len(query) > 2000:
            return False, "Query too long (max 2000 characters)"

        return True, ""

    @staticmethod
    def _validate_osb_status_message(message: Dict[str, Any]) -> tuple[bool, str]:
        """Validate OSB status message format."""
        required_fields = ["session_id", "status"]

        for field in required_fields:
            if field not in message:
                return False, f"Missing required field: {field}"

        valid_statuses = ["processing", "waiting", "complete", "error"]
        if message.get("status") not in valid_statuses:
            return False, f"Status must be one of: {valid_statuses}"

        return True, ""

    @staticmethod
    def _validate_osb_partial_message(message: Dict[str, Any]) -> tuple[bool, str]:
        """Validate OSB partial response message format."""
        required_fields = ["session_id", "agent", "partial_response"]

        for field in required_fields:
            if field not in message:
                return False, f"Missing required field: {field}"

        valid_agents = ["researcher", "policy_analyst", "fact_checker", "explorer"]
        if message.get("agent") not in valid_agents:
            return False, f"Agent must be one of: {valid_agents}"

        return True, ""

    @staticmethod
    def _validate_osb_complete_message(message: Dict[str, Any]) -> tuple[bool, str]:
        """Validate OSB complete response message format."""
        required_fields = ["session_id", "synthesis_summary", "agent_responses"]

        for field in required_fields:
            if field not in message:
                return False, f"Missing required field: {field}"

        agent_responses = message.get("agent_responses", {})
        if not isinstance(agent_responses, dict):
            return False, "agent_responses must be a dictionary"

        return True, ""

    @staticmethod
    def _validate_osb_error_message(message: Dict[str, Any]) -> tuple[bool, str]:
        """Validate OSB error message format."""
        required_fields = ["session_id", "error_type", "error_message"]

        for field in required_fields:
            if field not in message:
                return False, f"Missing required field: {field}"

        return True, ""


@asynccontextmanager
<<<<<<< HEAD
async def websocket_test_context(session_ids: List[str], base_uri: str) -> AsyncGenerator[List[WebSocketTestSession], None]:
=======
async def websocket_test_context(
    session_ids: List[str], base_uri: str
) -> AsyncGenerator[List[WebSocketTestSession], None]:
>>>>>>> origin/stable
    """Context manager for WebSocket test sessions with automatic cleanup."""
    sessions = []

    try:
        # Create and connect sessions
        for session_id in session_ids:
            session = WebSocketTestSession(session_id, f"{base_uri}/ws/{session_id}")
            await session.connect()
            sessions.append(session)

        yield sessions

    finally:
        # Cleanup all sessions
        disconnect_tasks = [session.disconnect() for session in sessions]
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)


<<<<<<< HEAD
async def simulate_osb_workflow(session: WebSocketTestSession, query: str, expected_agents: List[str] = None) -> Dict[str, Any]:
=======
async def simulate_osb_workflow(
    session: WebSocketTestSession, query: str, expected_agents: List[str] = None
) -> Dict[str, Any]:
>>>>>>> origin/stable
    """Simulate complete OSB workflow and validate responses."""
    if expected_agents is None:
        expected_agents = ["researcher", "policy_analyst", "fact_checker", "explorer"]

    workflow_results = {
        "query": query,
        "expected_agents": expected_agents,
        "responses_received": [],
        "agents_responded": set(),
        "workflow_complete": False,
        "error_occurred": False,
        "workflow_duration": 0,
    }

    start_time = time.time()

    try:
        # Send initial OSB query
        await session.send_osb_query(query, case_number="TEST-WORKFLOW-001")

        # Wait for responses
        timeout = 30.0  # 30 second timeout for workflow
        end_time = time.time() + timeout

        while time.time() < end_time:
            message = await session.receive_message(timeout=5.0)

            if message is None:
                continue

            workflow_results["responses_received"].append(message)

            message_type = message.get("type")

            if message_type == "osb_partial":
                agent = message.get("agent")
                if agent:
                    workflow_results["agents_responded"].add(agent)

            elif message_type == "osb_complete":
                workflow_results["workflow_complete"] = True
                break

            elif message_type == "osb_error":
                workflow_results["error_occurred"] = True
                break

        workflow_results["workflow_duration"] = time.time() - start_time

    except Exception as e:
        workflow_results["error_occurred"] = True
        workflow_results["error_message"] = str(e)
        workflow_results["workflow_duration"] = time.time() - start_time

    return workflow_results


# Utility functions for test setup and teardown


<<<<<<< HEAD
def validate_websocket_test_results(results: Dict[str, Any], expected_criteria: Dict[str, Any]) -> tuple[bool, List[str]]:
=======
def validate_websocket_test_results(
    results: Dict[str, Any], expected_criteria: Dict[str, Any]
) -> tuple[bool, List[str]]:
>>>>>>> origin/stable
    """Validate WebSocket test results against expected criteria."""
    validation_errors = []

    # Check connection success rate
    if "connection_success_rate" in expected_criteria:
        expected_rate = expected_criteria["connection_success_rate"]
        actual_rate = results.get("connection_success_rate", 0)
        if actual_rate < expected_rate:
<<<<<<< HEAD
            validation_errors.append(f"Connection success rate {actual_rate:.2f} below expected {expected_rate:.2f}")
=======
            validation_errors.append(
                f"Connection success rate {actual_rate:.2f} below expected {expected_rate:.2f}"
            )
>>>>>>> origin/stable

    # Check error rate
    if "max_error_rate" in expected_criteria:
        max_error_rate = expected_criteria["max_error_rate"]
        actual_error_rate = results.get("aggregate_metrics", {}).get("error_rate", 1.0)
        if actual_error_rate > max_error_rate:
<<<<<<< HEAD
            validation_errors.append(f"Error rate {actual_error_rate:.2f} exceeds maximum {max_error_rate:.2f}")
=======
            validation_errors.append(
                f"Error rate {actual_error_rate:.2f} exceeds maximum {max_error_rate:.2f}"
            )
>>>>>>> origin/stable

    # Check average connection time
    if "max_connection_time" in expected_criteria:
        max_connection_time = expected_criteria["max_connection_time"]
<<<<<<< HEAD
        actual_connection_time = results.get("aggregate_metrics", {}).get("average_connection_time", float("inf"))
        if actual_connection_time > max_connection_time:
            validation_errors.append(f"Average connection time {actual_connection_time:.3f}s exceeds maximum {max_connection_time:.3f}s")
=======
        actual_connection_time = results.get("aggregate_metrics", {}).get(
            "average_connection_time", float("inf")
        )
        if actual_connection_time > max_connection_time:
            validation_errors.append(
                f"Average connection time {actual_connection_time:.3f}s exceeds maximum {max_connection_time:.3f}s"
            )
>>>>>>> origin/stable

    # Check message throughput
    if "min_messages_per_second" in expected_criteria:
        min_throughput = expected_criteria["min_messages_per_second"]
<<<<<<< HEAD
        total_messages = results.get("aggregate_metrics", {}).get("total_messages_sent", 0)
        test_duration = results.get("test_duration", 1)
        actual_throughput = total_messages / test_duration
        if actual_throughput < min_throughput:
            validation_errors.append(f"Message throughput {actual_throughput:.2f} msg/s below minimum {min_throughput:.2f} msg/s")
=======
        total_messages = results.get("aggregate_metrics", {}).get(
            "total_messages_sent", 0
        )
        test_duration = results.get("test_duration", 1)
        actual_throughput = total_messages / test_duration
        if actual_throughput < min_throughput:
            validation_errors.append(
                f"Message throughput {actual_throughput:.2f} msg/s below minimum {min_throughput:.2f} msg/s"
            )
>>>>>>> origin/stable

    return len(validation_errors) == 0, validation_errors
