import asyncio
import importlib
import logging
import random
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import shortuuid
from fastapi import WebSocketDisconnect
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed


class SessionStatus(str, Enum):
    """Session status enumeration for robust lifecycle management."""

    INITIALIZING = "initializing"    # Session created, resources allocating
    ACTIVE = "active"               # Ready for operations
    PAUSED = "paused"              # Temporarily suspended (legacy)
    RECONNECTING = "reconnecting"   # Client disconnected, awaiting reconnect
    TERMINATING = "terminating"     # Cleanup in progress
    COMPLETED = "completed"         # Successfully completed
    TERMINATED = "terminated"       # Cleanup complete
    EXPIRED = "expired"            # Session expired due to timeout
    ERROR = "error"                # Failed state, needs manual cleanup
    FAILED = "failed"              # Legacy failed state


from buttermilk import logger
from buttermilk._core import (
    AgentTrace,
    logger,  # noqa
)
from buttermilk._core.agent import ErrorEvent
from buttermilk._core.context import set_logging_context
from buttermilk._core.contract import (
    ErrorEvent,
    FlowEvent,
    FlowMessage,
    UIMessage,
)
from buttermilk._core.exceptions import FatalError
from buttermilk._core.log import logger
from buttermilk._core.orchestrator import Orchestrator, OrchestratorProtocol
from buttermilk._core.types import Record, RunRequest
from buttermilk.api.job_queue import JobQueueClient
from buttermilk.api.services.data_service import DataService
from buttermilk.api.services.message_service import MessageService
from buttermilk.utils.utils import expand_dict


class SessionResources(BaseModel):
    """Tracks all resources allocated to a session for proper cleanup."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tasks: set[asyncio.Task] = Field(default_factory=set)
    websockets: set[Any] = Field(default_factory=set)
    file_handles: set[Any] = Field(default_factory=set)  # For IO objects
    memory_usage: int = Field(default=0)  # Bytes
    custom_resources: dict[str, Any] = Field(default_factory=dict)  # For extension

    def add_task(self, task: asyncio.Task) -> None:
        """Add a task to be tracked."""
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def add_websocket(self, websocket: Any) -> None:
        """Add a WebSocket to be tracked."""
        self.websockets.add(websocket)

    def add_file_handle(self, file_handle: Any) -> None:
        """Add a file handle to be tracked."""
        self.file_handles.add(file_handle)

    def add_custom_resource(self, name: str, resource: Any) -> None:
        """Add a custom resource to be tracked."""
        self.custom_resources[name] = resource

    async def cleanup(self) -> dict[str, Any]:
        """Cleanup all tracked resources and return a report."""
        report = {
            "tasks_cancelled": 0,
            "websockets_closed": 0,
            "files_closed": 0,
            "custom_cleaned": 0,
            "errors": []
        }

        # Cancel tasks
        for task in list(self.tasks):
            if not task.done():
                task.cancel()
                report["tasks_cancelled"] += 1

        if self.tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                report["errors"].append("Timeout waiting for task cancellation")

            self.tasks.clear()

        # Close WebSockets
        for ws in list(self.websockets):
            try:
                if hasattr(ws, "close"):
                    await ws.close()
                    report["websockets_closed"] += 1
            except Exception as e:
                report["errors"].append(f"Error closing WebSocket: {e}")
        self.websockets.clear()

        # Close file handles
        for fh in list(self.file_handles):
            try:
                if hasattr(fh, "close"):
                    fh.close()
                    report["files_closed"] += 1
            except Exception as e:
                report["errors"].append(f"Error closing file handle: {e}")
        self.file_handles.clear()

        # Cleanup custom resources
        for name, resource in list(self.custom_resources.items()):
            try:
                if hasattr(resource, "cleanup"):
                    cleanup_result = resource.cleanup()
                    if asyncio.iscoroutine(cleanup_result):
                        await cleanup_result
                elif hasattr(resource, "close"):
                    close_result = resource.close()
                    if asyncio.iscoroutine(close_result):
                        await close_result
                report["custom_cleaned"] += 1
            except Exception as e:
                report["errors"].append(f"Error cleaning up {name}: {e}")
        self.custom_resources.clear()

        return report


class FlowRunContext(BaseModel):
    """Encapsulates all state for a single flow run with session management."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    flow_name: str = ""
    flow_task: Any | None = None
    orchestrator: Orchestrator | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    session_id: str
    callback_to_groupchat: Any = None
    messages: list = []
    progress: dict = Field(default_factory=dict)

    # Session management fields
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(UTC))
    background_tasks: set[asyncio.Task] = Field(default_factory=set)  # DEPRECATED: Use resources.tasks
    session_timeout: int = 3600  # 1 hour default timeout in seconds
    resources: SessionResources = Field(default_factory=SessionResources)  # Resource tracking

    websocket: Any = None

    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.now(UTC)

    def is_expired(self) -> bool:
        """Check if the session has expired based on timeout."""
        return (datetime.now(UTC) - self.last_activity).total_seconds() > self.session_timeout

    def get_isolated_topic(self, base_topic: str) -> str:
        """Generate session-isolated topic names for message routing."""
        return f"{self.session_id}:{base_topic}"

    def add_task(self, task: asyncio.Task) -> None:
        """Add a task to be tracked by this session."""
        self.resources.add_task(task)
        self.update_activity()

    def add_websocket(self, websocket: Any) -> None:
        """Add a WebSocket to be tracked by this session."""
        self.resources.add_websocket(websocket)
        self.websocket = websocket  # Maintain backward compatibility
        self.update_activity()

    def add_file_handle(self, file_handle: Any) -> None:
        """Add a file handle to be tracked by this session."""
        self.resources.add_file_handle(file_handle)
        self.update_activity()

    def add_custom_resource(self, name: str, resource: Any) -> None:
        """Add a custom resource to be tracked by this session."""
        self.resources.add_custom_resource(name, resource)
        self.update_activity()

    async def cleanup(self) -> None:
        """Clean up session resources with timeout and verification."""
        logger.debug(f"Starting cleanup for session {self.session_id}")

        try:
            # Set status to terminating (Phase 2 enhancement)
            self.status = SessionStatus.TERMINATING

            # Add flow task to resource tracker if it exists
            if self.flow_task and not self.flow_task.done():
                self.resources.add_task(self.flow_task)

            # Add any legacy background tasks to resource tracker
            for task in self.background_tasks:
                self.resources.add_task(task)
            self.background_tasks.clear()  # Clear the legacy set

            # Add WebSocket to resource tracker
            if self.websocket:
                self.resources.add_websocket(self.websocket)

            # Add orchestrator to custom resources if it exists
            if self.orchestrator:
                self.resources.add_custom_resource("orchestrator", self.orchestrator)

            # Perform comprehensive resource cleanup
            logger.debug(f"Cleaning up resources for session {self.session_id}")
            cleanup_report = await self.resources.cleanup()

            # Log cleanup report
            if cleanup_report.get("errors"):
                logger.warning(f"Session {self.session_id} cleanup completed with errors: {cleanup_report}")
                self.status = SessionStatus.ERROR
            else:
                logger.info(f"Session {self.session_id} cleaned up successfully: {cleanup_report}")
                self.status = SessionStatus.TERMINATED

        except Exception as e:
            logger.error(f"Error during session cleanup for {self.session_id}: {e}")
            # Ensure status is set even if cleanup fails
            self.status = SessionStatus.ERROR

    async def monitor_ui(self) -> AsyncGenerator[RunRequest, None]:
        """Monitor the UI for incoming messages."""
        while True:
            await asyncio.sleep(0.1)

            # Check if the WebSocket is connected
            if not self.websocket or self.websocket.client_state != WebSocketState.CONNECTED:
                logger.debug(f"WebSocket not connected for session {self.session_id}")
                continue

            try:
                data = await self.websocket.receive_json()
                self.update_activity()  # Update activity timestamp on message

                message = await MessageService.process_message_from_ui(data)
                if not message:
                    logger.debug(f"No message returned from process_message_from_ui for data: {data}")
                    continue

                if isinstance(message, RunRequest):
                    # Generate a request to run the flow with the new parameters
                    message.callback_to_ui = self.send_message_to_ui
                    message.session_id = self.session_id
                    logger.info(f"Yielding RunRequest for flow '{message.flow}' in session {self.session_id}")

                    yield message
                elif not self.callback_to_groupchat:
                    # Group chat has not started yet
                    logger.debug(f"Group chat not yet started for session {self.session_id}")
                    continue
                else:
                    await self.callback_to_groupchat(message)

            except WebSocketDisconnect:
                logger.info(f"Client {self.session_id} disconnected.")
                self.websocket = None
                # Don't break immediately - let the session manager handle reconnection
                break
            except Exception as e:
                logger.error(f"Error receiving/processing client message for {self.session_id}: {e}")
                self.websocket = None
                break
                # raise FatalError(f"Error receiving/processing client message for {self.session_id}: {e}")

    async def send_message_to_ui(self, message: AgentTrace | UIMessage | Record | FlowEvent | FlowMessage) -> None:
        """Send a message to a WebSocket connection.

        Args:
            message: The message to send

        """
        formatted_message = MessageService.format_message_for_client(message)
        if not formatted_message:
            logger.debug(f"Dropping message not handled by client: {message}")
            return

        try:
            message_type = formatted_message.type
            message_data_to_send = formatted_message.model_dump(mode="json", exclude_unset=True, exclude_none=True)

            logger.debug(f"[FlowRunner.send_message_to_ui] Sending message of type {message_type} to UI for session {self.session_id}")

            @retry(
                stop=stop_after_attempt(10),
                wait=wait_fixed(3),  # Wait 3 seconds between attempts
                retry=retry_if_exception_type(Exception),  # Retry on any exception during send
                reraise=True,  # Reraise the last exception if all retries fail
                # before_sleep=before_sleep_log(logger, logging.WARNING),  # Log a warning before retrying
            )
            async def _send_with_retry_internal():
                if not self.websocket:
                    # Raise an error to be caught by tenacity or the outer try/except
                    raise RuntimeError(f"WebSocket is None for session {self.session_id} during send attempt.")

                if self.websocket.client_state != WebSocketState.CONNECTED:
                    logger.debug(
                        f"WebSocket not connected (state: {self.websocket.client_state}) for session {self.session_id}. Retrying send.",
                    )
                    # Proceed to send; if it fails due to state, tenacity will catch and retry.

                await self.websocket.send_json(message_data_to_send)
                logger.debug(f"Message sent to UI for session {self.session_id}: {message_type}")

            await _send_with_retry_internal()

        except Exception as e:
            # Attempt to send an error message back to the client if the websocket is still viable
            if self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
                try:
                    error_event = ErrorEvent(source="websocket_manager", content=f"Failed to send message to client: {e!s}")
                    error_message_data = {"content": error_event.model_dump(), "type": "system_message"}
                    await self.websocket.send_json(error_message_data)
                except Exception as e_fallback:
                    pass

            logger.warning(f"Cannot send error notification to UI for session {self.session_id}: WebSocket not available or not connected. {e}")


class OrchestratorFactory:
    """Factory for creating and managing orchestrator instances with proper lifecycle."""

    @staticmethod
    def create_orchestrator(flow_config: OrchestratorProtocol, flow_name: str) -> Orchestrator:
        """Create a completely fresh orchestrator instance.

        Args:
            flow_config: The flow configuration to use
            flow_name: The name of the flow (for error reporting)

        Returns:
            A new orchestrator instance with fresh state

        Raises:
            ValueError: If orchestrator class cannot be found or instantiated

        """
        try:
            # Extract orchestrator class path
            orchestrator_path = flow_config.orchestrator
            module_name, class_name = orchestrator_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            orchestrator_cls = getattr(module, class_name)

            # Create a fresh config copy to avoid shared state
            if hasattr(flow_config, "model_dump"):
                config = flow_config.model_dump()
            else:
                # Convert OmegaConf objects to standard Python types recursively
                from buttermilk.utils.validators import convert_omegaconf_objects
                config = convert_omegaconf_objects(dict(flow_config))

            # Create and return a fresh instance
            orchestrator = orchestrator_cls(**config)

            logger.debug(f"Created fresh orchestrator for flow '{flow_name}': {orchestrator_cls.__name__}")
            return orchestrator

        except Exception as e:
            raise ValueError(f"Failed to create orchestrator for flow '{flow_name}': {e}") from e

    @staticmethod
    async def cleanup_orchestrator(orchestrator: Orchestrator) -> None:
        """Clean up an orchestrator instance and its resources.

        Args:
            orchestrator: The orchestrator to clean up

        """
        cleanup_method = getattr(orchestrator, "cleanup", None)
        if cleanup_method and callable(cleanup_method):
            try:
                result = cleanup_method()
                if asyncio.iscoroutine(result):
                    await result
                logger.debug(f"Orchestrator cleanup completed: {orchestrator.__class__.__name__}")
            except Exception as e:
                logger.warning(f"Error during orchestrator cleanup: {e}")


class SessionManager:
    """Manages session lifecycle, timeouts, and cleanup with atomic operations."""

    def __init__(self, session_timeout: int = 3600):
        self.sessions: dict[str, FlowRunContext] = {}
        self.session_timeout = session_timeout
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

        # Atomic operation support
        self.session_locks: dict[str, asyncio.Lock] = {}  # Prevent race conditions
        self.session_resources: dict[str, SessionResources] = {}  # Track all resources
        self.active_connections: dict[str, set[Any]] = {}  # Multiple connections per session
        self.shutdown_handlers: dict[str, Callable] = {}  # Custom cleanup per session
        self._global_lock = asyncio.Lock()  # Protect session creation/deletion

    async def start(self) -> None:
        """Start the session manager and background cleanup task."""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Session manager started with background cleanup")

    async def stop(self) -> None:
        """Stop the session manager and cleanup all sessions."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Clean up all remaining sessions
        for session_id in list(self.sessions.keys()):
            await self.cleanup_session(session_id)

        logger.info("Session manager stopped and all sessions cleaned up")

    async def get_or_create_session(self, session_id: str, websocket: Any = None) -> FlowRunContext:
        """Get existing session or create a new one with atomic operations.

        Args:
            session_id: Unique identifier for the session
            websocket: Optional WebSocket connection

        Returns:
            The session context

        """
        async with self._global_lock:
            # Check if session exists
            if session_id in self.sessions:
                session = self.sessions[session_id]
                # Add websocket to connection pool if provided
                if websocket:
                    if session_id not in self.active_connections:
                        self.active_connections[session_id] = set()
                    self.active_connections[session_id].add(websocket)
                    session.websocket = websocket
                    session.update_activity()
                    logger.debug(f"Added WebSocket to existing session {session_id}")
                return session

            # Create new session with INITIALIZING status
            session = FlowRunContext(
                session_id=session_id,
                websocket=websocket,
                session_timeout=self.session_timeout,
                status=SessionStatus.INITIALIZING
            )

            # Initialize session infrastructure
            self.sessions[session_id] = session
            self.session_locks[session_id] = asyncio.Lock()
            self.session_resources[session_id] = SessionResources()
            self.active_connections[session_id] = set()

            if websocket:
                self.active_connections[session_id].add(websocket)
                session.add_websocket(websocket)

            logger.info(f"Created new session {session_id} with INITIALIZING status")

            # Transition to ACTIVE after initialization
            await self._transition_session_status(session_id, SessionStatus.ACTIVE)

            return session

    async def _transition_session_status(self, session_id: str, new_status: SessionStatus) -> bool:
        """Safely transition a session to a new status.
        
        Args:
            session_id: The session to transition
            new_status: The new status to transition to
            
        Returns:
            True if transition was successful, False otherwise
        """
        if session_id not in self.sessions:
            logger.warning(f"Attempted to transition status for non-existent session {session_id}")
            return False

        session = self.sessions[session_id]
        old_status = session.status

        # Validate transition (enhanced validation for Phase 2)
        valid_transitions = {
            SessionStatus.INITIALIZING: [SessionStatus.ACTIVE, SessionStatus.ERROR],
            SessionStatus.ACTIVE: [SessionStatus.TERMINATING, SessionStatus.RECONNECTING, SessionStatus.COMPLETED, SessionStatus.ERROR, SessionStatus.EXPIRED],
            SessionStatus.RECONNECTING: [SessionStatus.ACTIVE, SessionStatus.TERMINATING, SessionStatus.ERROR],
            SessionStatus.TERMINATING: [SessionStatus.TERMINATED, SessionStatus.ERROR],
            SessionStatus.COMPLETED: [SessionStatus.TERMINATED],
            SessionStatus.TERMINATED: [],  # Terminal state - no transitions allowed
            SessionStatus.ERROR: [SessionStatus.TERMINATING, SessionStatus.TERMINATED],
            SessionStatus.EXPIRED: [SessionStatus.TERMINATING, SessionStatus.TERMINATED],
            SessionStatus.PAUSED: [SessionStatus.ACTIVE, SessionStatus.TERMINATING, SessionStatus.ERROR],  # Legacy support
            SessionStatus.FAILED: [SessionStatus.TERMINATING, SessionStatus.TERMINATED],  # Legacy support
        }

        if old_status in valid_transitions and new_status not in valid_transitions[old_status]:
            logger.warning(f"Invalid status transition for session {session_id}: {old_status} -> {new_status}")
            return False

        session.status = new_status
        logger.debug(f"Session {session_id} status: {old_status} -> {new_status}")
        return True

    async def cleanup_session(self, session_id: str) -> bool:
        """Clean up and remove a session with atomic operations."""

        async with self._global_lock:
            if session_id not in self.sessions:
                return False

            # Transition to TERMINATING status
            await self._transition_session_status(session_id, SessionStatus.TERMINATING)

            session = self.sessions[session_id]

            # Execute custom shutdown handler if one exists
            if session_id in self.shutdown_handlers:
                try:
                    await self.shutdown_handlers[session_id]()
                except Exception as e:
                    logger.warning(f"Error in custom shutdown handler for session {session_id}: {e}")
                finally:
                    del self.shutdown_handlers[session_id]

            # Perform session cleanup
            await session.cleanup()

            # Clean up session manager resources
            if session_id in self.session_locks:
                del self.session_locks[session_id]
            if session_id in self.session_resources:
                await self.session_resources[session_id].cleanup()
                del self.session_resources[session_id]
            if session_id in self.active_connections:
                self.active_connections[session_id].clear()
                del self.active_connections[session_id]

            # Remove session and transition to TERMINATED
            del self.sessions[session_id]
            logger.info(f"Session {session_id} removed and cleaned up (TERMINATED)")
            return True

    async def register_shutdown_handler(self, session_id: str, handler: Callable) -> None:
        """Register a custom shutdown handler for a session.
        
        Args:
            session_id: The session to register the handler for
            handler: An async callable that performs custom cleanup
        """
        self.shutdown_handlers[session_id] = handler
        logger.debug(f"Registered shutdown handler for session {session_id}")

    async def validate_session(self, session_id: str) -> bool:
        """Validate that a session is in a good state.
        
        Args:
            session_id: The session to validate
            
        Returns:
            True if session is valid, False otherwise
        """
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        # Check if session is in a valid operational state
        if session.status in [SessionStatus.TERMINATED, SessionStatus.ERROR]:
            return False

        # Check if session has expired
        if session.is_expired():
            logger.info(f"Session {session_id} has expired")
            await self._transition_session_status(session_id, SessionStatus.EXPIRED)
            return False

        # Additional health checks could be added here
        return True

    async def get_session_health(self, session_id: str) -> dict[str, Any]:
        """Get health information for a session.
        
        Args:
            session_id: The session to check
            
        Returns:
            Dictionary containing health information
        """
        if session_id not in self.sessions:
            return {"status": "not_found", "health": "unknown"}

        session = self.sessions[session_id]

        health_info = {
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "is_expired": session.is_expired(),
            "active_connections": len(self.active_connections.get(session_id, set())),
            "active_tasks": len(session.resources.tasks),
            "websockets": len(session.resources.websockets),
            "memory_usage": session.resources.memory_usage,
        }

        # Add orchestrator status if available
        if session.orchestrator:
            health_info["orchestrator_type"] = session.orchestrator.__class__.__name__
            health_info["has_orchestrator"] = True
        else:
            health_info["has_orchestrator"] = False

        return health_info

    async def handle_client_disconnect(self, session_id: str) -> bool:
        """Handle client disconnect by transitioning to RECONNECTING status instead of immediate cleanup.
        
        Args:
            session_id: The session that disconnected
            
        Returns:
            True if session was transitioned to RECONNECTING, False if session was cleaned up
        """
        if session_id not in self.sessions:
            logger.warning(f"Attempted to handle disconnect for non-existent session {session_id}")
            return False

        session = self.sessions[session_id]
        
        # Only allow reconnection for ACTIVE sessions
        if session.status != SessionStatus.ACTIVE:
            logger.info(f"Session {session_id} in status {session.status.value} - cleaning up instead of allowing reconnection")
            await self.cleanup_session(session_id)
            return False

        # Transition to RECONNECTING status
        success = await self._transition_session_status(session_id, SessionStatus.RECONNECTING)
        if success:
            # Clear the websocket but keep the session alive
            session.websocket = None
            # Clear active connections for this session
            if session_id in self.active_connections:
                self.active_connections[session_id].clear()
            
            logger.info(f"Session {session_id} transitioned to RECONNECTING - client can reconnect within {session.session_timeout}s")
            return True
        else:
            # Transition failed, clean up the session
            await self.cleanup_session(session_id)
            return False

    async def reconnect_session(self, session_id: str, websocket: Any) -> FlowRunContext | None:
        """Reconnect a client to an existing session in RECONNECTING status.
        
        Args:
            session_id: The session to reconnect to
            websocket: The new WebSocket connection
            
        Returns:
            The session if reconnection was successful, None otherwise
        """
        if session_id not in self.sessions:
            logger.warning(f"Attempted to reconnect to non-existent session {session_id}")
            return None

        session = self.sessions[session_id]
        
        # Only allow reconnection to sessions in RECONNECTING status
        if session.status != SessionStatus.RECONNECTING:
            logger.warning(f"Attempted to reconnect to session {session_id} in status {session.status.value}")
            return None

        # Check if session has expired
        if session.is_expired():
            logger.info(f"Session {session_id} has expired - cleaning up instead of reconnecting")
            await self.cleanup_session(session_id)
            return None

        # Reconnect the session
        session.websocket = websocket
        session.add_websocket(websocket)
        session.update_activity()
        
        # Add to active connections
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)

        # Transition back to ACTIVE status
        success = await self._transition_session_status(session_id, SessionStatus.ACTIVE)
        if success:
            logger.info(f"Successfully reconnected session {session_id}")
            return session
        else:
            logger.error(f"Failed to transition session {session_id} back to ACTIVE after reconnection")
            return None

    async def _periodic_cleanup(self) -> None:
        """Background task that periodically cleans up expired sessions with enhanced logic."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                cleanup_candidates = []

                # Use a copy of sessions to avoid modification during iteration
                sessions_snapshot = dict(self.sessions)

                for session_id, session in sessions_snapshot.items():
                    # Check for various cleanup conditions
                    if session.is_expired():
                        cleanup_candidates.append((session_id, "expired"))
                    elif session.status == SessionStatus.ERROR:
                        cleanup_candidates.append((session_id, "error_state"))
                    elif session.status == SessionStatus.COMPLETED:
                        # Clean up completed sessions after a grace period
                        time_since_completion = (datetime.now(UTC) - session.last_activity).total_seconds()
                        if time_since_completion > 300:  # 5 minutes grace period
                            cleanup_candidates.append((session_id, "completed_gracetime"))
                    elif session.status == SessionStatus.RECONNECTING:
                        # Clean up RECONNECTING sessions that have exceeded timeout
                        if session.is_expired():
                            cleanup_candidates.append((session_id, "reconnect_timeout"))
                    elif len(self.active_connections.get(session_id, set())) == 0:
                        # No active connections for extended period
                        time_since_activity = (datetime.now(UTC) - session.last_activity).total_seconds()
                        if time_since_activity > 1800:  # 30 minutes without connections
                            cleanup_candidates.append((session_id, "no_connections"))

                # Perform cleanup
                for session_id, reason in cleanup_candidates:
                    logger.info(f"Cleaning up session {session_id} (reason: {reason})")
                    await self.cleanup_session(session_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup task: {e}")


class FlowRunner(BaseModel):
    """Centralized service for running flows across different entry points.

    Handles orchestrator instantiation and execution in a consistent way, regardless
    of whether the flow is started from CLI, API, Slackbot, or Pub/Sub.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    flows: dict[str, OrchestratorProtocol]

    tasks: list = Field(default=[])
    mode: str = Field(default="api")
    ui: str = Field(default="console")
    human_in_loop: bool = False
    sessions: dict[str, FlowRunContext] = Field(default_factory=dict)  # Dictionary of active sessions (DEPRECATED)

    # New session management
    session_manager: SessionManager = Field(default_factory=lambda: SessionManager())
    _session_manager_started: bool = False

    # I think we comment this out because in flowrunner, we deal with Configs, not instantiated orchestrators. We instantiate later.
    # @field_validator("flows", mode="before")
    # @classmethod
    # def validate_flows(cls, v):
    #     """Convert OmegaConf flow configurations to Orchestrator instances."""
    #     if not v:
    #         return v

    #     from buttermilk._core.orchestrator import OrchestratorProtocol
    #     from omegaconf import DictConfig, OmegaConf

    #     converted_flows = {}
    #     for flow_name, flow_config in v.items():

    #         orchestrator_class_path = flow_config.orchestrator
    #         if orchestrator_class_path:
    #             module_path, class_name = orchestrator_class_path.rsplit(".", 1)
    #             import importlib

    #             module = importlib.import_module(module_path)
    #             orchestrator_class = getattr(module, class_name)
    #             converted_flows[flow_name] = orchestrator_class.model_validate(flow_config)
    #         else:
    #             converted_flows[flow_name] = Orchestrator.model_validate(flow_config)
    #     return converted_flows

    async def _ensure_session_manager_started(self) -> None:
        """Ensure the session manager is started."""
        if not self._session_manager_started:
            await self.session_manager.start()
            self._session_manager_started = True

    async def get_websocket_session_async(self, session_id: str, websocket: Any | None = None) -> FlowRunContext | None:
        """Get or create a session for the given session ID, handling reconnection scenarios.

        Args:
            session_id: Unique identifier for the session
            websocket: WebSocket connection for this session

        Returns:
            A FlowRunContext object representing the session

        """
        await self._ensure_session_manager_started()

        # Check if this is a reconnection to an existing session
        if session_id in self.session_manager.sessions:
            existing_session = self.session_manager.sessions[session_id]
            
            # If session is in RECONNECTING status, attempt to reconnect
            if existing_session.status == SessionStatus.RECONNECTING and websocket:
                logger.info(f"Attempting to reconnect to session {session_id}")
                reconnected_session = await self.session_manager.reconnect_session(session_id, websocket)
                if reconnected_session:
                    return reconnected_session
                else:
                    # Reconnection failed, fall through to create new session
                    logger.warning(f"Failed to reconnect to session {session_id}, creating new session")
            elif existing_session.status in [SessionStatus.ACTIVE, SessionStatus.INITIALIZING]:
                # Session is already active, just add the websocket
                if websocket:
                    existing_session.add_websocket(websocket)
                return existing_session

        # Create new session if no websocket provided or reconnection failed
        if not websocket:
            return None

        session = await self.session_manager.get_or_create_session(session_id, websocket)
        return session

    async def cleanup(self) -> None:
        """Clean up the FlowRunner and all its sessions."""
        if self._session_manager_started:
            await self.session_manager.stop()
            self._session_manager_started = False

    async def pull_and_run_task(self) -> None:
        """Pull tasks from the queue and run them."""
        # Initialize the queue_manager if needed
        queue_manager = getattr(self, "queue_manager", None)
        if queue_manager is None:
            self.queue_manager = JobQueueClient()

        # Pull task from the queue
        request = await self.queue_manager.pull_single_task()

        raise FatalError("Need to create the sssion object first")
        if request:
            # Run the task with a fresh orchestrator
            logger.info(f"Running task from queue: {request.flow} (Task ID: {request.job_id})")  # Updated message
            await self.run_flow(request, wait_for_completion=True)
        else:
            logger.debug("No tasks available in the queue")

    def _create_fresh_orchestrator(self, flow_name: str) -> Orchestrator:
        """Create a completely fresh orchestrator instance using the factory.

        Args:
            flow_name: The name of the flow to create an orchestrator for

        Returns:
            A new orchestrator instance with fresh state

        Raises:
            ValueError: If flow_name doesn't exist in flows

        """
        if flow_name not in self.flows:
            raise ValueError(f"Flow '{flow_name}' not found. Available flows: {list(self.flows.keys())}")

        flow_config = self.flows[flow_name]
        return OrchestratorFactory.create_orchestrator(flow_config, flow_name)

    async def _cleanup_flow_context(self, context: FlowRunContext) -> None:
        """Clean up resources associated with a flow run.

        Args:
            context: The flow run context to clean up

        """
        # Use the enhanced cleanup from FlowRunContext
        await context.cleanup()

    async def run_flow(self,
                      run_request: RunRequest,
                      wait_for_completion: bool = False,
                      **kwargs) -> None:
        """Run a flow based on its configuration and a request.

        Args:
            run_request: The request containing input parameters
            wait_for_completion: If True, await the flow's completion before returning.
                                 If False (default), start the flow as a background
                                 task and return immediately.
            history: Optional conversation history (for chat-based interfaces)
            **kwargs: Additional keyword arguments for orchestrator instantiation

        Returns:
            If wait_for_completion is True, returns the result of the orchestrator run.
            If wait_for_completion is False, returns a callback function.

        Raises:
            ValueError: If orchestrator isn't specified or unknown

        """
        # Ensure BM is fully initialized before running flow
        from buttermilk._core.dmrc import get_bm
        bm = get_bm()
        if hasattr(bm, 'ensure_initialized'):
            await bm.ensure_initialized()
            logger.debug("BM initialization verified before flow execution")
        
        # Initialize metrics tracking
        import time
        start_time = time.time()
        success = False

        try:
            from buttermilk.monitoring import get_metrics_collector
            metrics_collector = get_metrics_collector()
        except Exception as e:
            logger.debug(f"Failed to initialize metrics collector: {e}")
            metrics_collector = None

        # Ensure session manager is started
        await self._ensure_session_manager_started()

        # Create a fresh orchestrator instance
        fresh_orchestrator = self._create_fresh_orchestrator(run_request.flow)

        # set a high max callback duration when dealing with LLMs
        asyncio.get_event_loop().slow_callback_duration = 120

        # Get or create session using the session manager
        _session = await self.session_manager.get_or_create_session(run_request.session_id)

        set_logging_context(run_request.session_id)
        _session.flow_name = run_request.flow
        _session.orchestrator = fresh_orchestrator
        _session.callback_to_groupchat = fresh_orchestrator.make_publish_callback()
        _session.update_activity()  # Update activity timestamp

        # Create the task and register it with the session
        _session.flow_task = asyncio.create_task(fresh_orchestrator.run(request=run_request))  # type: ignore
        _session.add_task(_session.flow_task)

        # ======== MAJOR EVENT: FLOW STARTING ========
        # Log detailed information about flow start
        logger.info(f"🚀 FLOW STARTING: '{run_request.flow}' (ID: {run_request.job_id}).\n📋 RunRequest: {run_request.model_dump_json(indent=2)}\n⚙️ Source: {', '.join(run_request.source) if run_request.source else 'direct'}\n✅ New flow instance created - all state has been reset")

        try:
            if wait_for_completion:
                # Wait for the task
                await _session.flow_task
                await self.session_manager._transition_session_status(run_request.session_id, SessionStatus.COMPLETED)
                success = True
                return
        except Exception as e:
            await self.session_manager._transition_session_status(run_request.session_id, SessionStatus.ERROR)
            logger.error(f"Error running flow '{run_request.flow}': {e}")
            raise
        finally:
            # Record flow execution metrics
            if metrics_collector:
                execution_time = time.time() - start_time
                try:
                    metrics_collector.record_flow_execution(
                        flow_name=run_request.flow,
                        execution_time=execution_time,
                        success=success
                    )
                except Exception as e:
                    logger.debug(f"Failed to record flow execution metrics: {e}")

            if wait_for_completion:
                # Clean up after completion if we were waiting
                await self.session_manager.cleanup_session(run_request.session_id)
        return

    async def create_batch(self, flow_name, max_records: int | None) -> list[RunRequest]:
        """Create a new batch job from the given request.

        Args:
            batch_request: The batch configuration

        Returns:
            The created batch metadata

        Raises:
            ValueError: If the flow doesn't exist or record extraction fails

        """
        # Extract record IDs from the flow's data source
        record_ids = await DataService.get_records_for_flow(flow_name=flow_name, flow_runner=self)
        logger.info(f"Extracted {len(record_ids)} record IDs for flow '{flow_name}'")

        # Create multiple iterations by multiplying the parameters
        iteration_values = expand_dict(self.flows[flow_name].parameters) or [{}]
        logger.debug(f"Expanded {len(self.flows[flow_name].parameters)} parameters for batch into {len(iteration_values)} variants")

        # Shuffle records
        random.shuffle(record_ids)
        logger.debug(f"Shuffled {len(record_ids)} record IDs")

        batch_id = str(shortuuid.uuid())

        # Create run requests for each record and parameter combination
        job_definitions = []

        # Apply iteration values
        for iteration_params in iteration_values:
            for i, record in enumerate(record_ids):
                job = RunRequest(ui_type="batch",
                    batch_id=batch_id,
                    flow=flow_name,
                    record_id=record["record_id"],
                    parameters=iteration_params, callback_to_ui=None,
                )
                job_definitions.append(job)
                logger.debug(f"Created run request: {job.model_dump_json()}")

                # Apply max_records limit if specified
                if max_records is not None and max_records > 0 and i >= max_records:
                    break

        if max_records is not None and max_records > 0:
            logger.info(f"Limited to {max_records} record IDs, returning {len(iteration_values)} iterations for {len(job_definitions)} jobs total.")
        else:
            logger.info(f"Returning {len(iteration_values)} iterations for {len(job_definitions)} jobs total.")

        random.shuffle(job_definitions)

        try:
            # Enqueue the batch for processing
            job_queue = JobQueueClient()

            for request in job_definitions:
                job_queue.publish_job(request)

        except Exception as e:
            msg = f"Failed to publish job to queue: {e}"
            raise FatalError(msg) from e

        return job_definitions

    async def run_batch_job(self, callback_to_ui: Callable, max_jobs: int = 1, wait_for_completion: bool = True) -> None:
        """Pull and run jobs from the queue, ensuring fresh state for each job.

        Args:
            max_jobs: Maximum number of jobs to process in this batch run

        Raises:
            FatalError: If no run requests are found in the queue
            Exception: If there's an error running a job

        """
        try:
            worker = JobQueueClient(
                max_concurrent_jobs=1,  # Process one job at a time to maintain isolation
            )

            jobs_processed = 0

            while jobs_processed < max_jobs:
                # Pull a job from the queue
                run_request = await worker.pull_single_task()
                if not run_request:
                    if jobs_processed == 0:
                        # Only raise an error if we didn't process any jobs
                        raise FatalError("No run request found in the queue.")
                    break  # No more jobs to process

                run_request.callback_to_ui = callback_to_ui

                logger.info(f"Processing batch job {jobs_processed + 1}/{max_jobs}: {run_request.flow} (Job ID: {run_request.job_id})")
                try:
                    await self.run_flow(run_request=run_request, wait_for_completion=wait_for_completion)
                    if wait_for_completion:
                        logger.info(f"Successfully completed job {run_request.job_id}")
                    else:
                        logger.info(f"Job {run_request.job_id} started in the background")
                except Exception as job_error:
                    logger.error(f"Error running job {run_request.job_id}: {job_error}")
                    # Continue processing other jobs even if one fails

                jobs_processed += 1

            logger.info(f"Batch processing complete. Processed {jobs_processed} jobs.")

        except FatalError:
            # Re-raise FatalError to be handled by the caller
            raise
        except Exception as e:
            logger.error(f"Fatal error during batch processing: {e}")
            raise