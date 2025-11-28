import asyncio
import importlib
import json
import random
import time
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import shortuuid
from pydantic import BaseModel, ConfigDict, Field

from buttermilk import ExecutionTrace, logger

# Starlette is optional - only needed for WebSocket API mode
try:
    from starlette.websockets import WebSocketDisconnect
except ImportError:
    # Create a placeholder exception that will never match if starlette isn't installed
    class WebSocketDisconnect(Exception):  # type: ignore[no-redef]
        """Placeholder for starlette.websockets.WebSocketDisconnect when starlette is not installed."""

        pass
from buttermilk._core.context import set_logging_context
from buttermilk._core.contract import (
    ErrorEvent,
    FlowEvent,
    FlowMessage,
    SystemPromptMessage,
)
from buttermilk._core.exceptions import FatalError
from buttermilk._core.orchestrator import Orchestrator, OrchestratorProtocol
from buttermilk._core.types import ProcessingSummary, Record, RunRequest
from buttermilk.api.job_queue import JobQueueClient
from buttermilk.api.services.message_service import MessageService
from buttermilk.api.services.session_storage import SessionStorageService
from buttermilk.utils import scrub_serializable
from buttermilk.utils.otel import (
    attach_session_baggage,
    detach_session_baggage,
    start_root_span,
    # Session root span functions removed (Phase 1 OTEL fix)
    # end_session_root_span,
    # start_session_root_span,
)
from buttermilk.utils.utils import expand_dict


class SessionStatus(str, Enum):
    """Session status enumeration for robust lifecycle management."""

    INITIALIZING = "initializing"  # Session created, resources allocating
    ACTIVE = "active"  # Ready for operations
    PAUSED = "paused"  # Temporarily suspended (legacy)
    RECONNECTING = "reconnecting"  # Client disconnected, awaiting reconnect
    TERMINATING = "terminating"  # Cleanup in progress
    COMPLETED = "completed"  # Successfully completed
    TERMINATED = "terminated"  # Cleanup complete
    EXPIRED = "expired"  # Session expired due to timeout
    ERROR = "error"  # Failed state, needs manual cleanup
    FAILED = "failed"  # Legacy failed state


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

    async def cleanup(self) -> dict[str, Any]:  # noqa: PLR0912
        """Cleanup all tracked resources and return a report."""
        report = {
            "tasks_cancelled": 0,
            "websockets_closed": 0,
            "files_closed": 0,
            "custom_cleaned": 0,
            "errors": [],
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
                    timeout=5.0,
                )
            except TimeoutError:
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
    background_tasks: set[asyncio.Task] = Field(
        default_factory=set
    )  # DEPRECATED: Use resources.tasks
    session_timeout: int = 3600  # 1 hour default timeout in seconds
    resources: SessionResources = Field(
        default_factory=SessionResources
    )  # Resource tracking

    websocket: Any = None
    monitor_ui_task: asyncio.Task | None = None  # Track active monitor_ui task
    # Telemetry baggage token for session-level context propagation
    _otel_baggage_token: Any | None = None
    _otel_session_root: tuple[Any, Any] | None = None  # (span, token)

    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.now(UTC)

    def is_expired(self) -> bool:
        """Check if the session has expired based on timeout."""
        return (
            datetime.now(UTC) - self.last_activity
        ).total_seconds() > self.session_timeout

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

    def cancel_monitor_ui_task(self) -> None:
        """Cancel the active monitor_ui task if it exists."""
        if self.monitor_ui_task and not self.monitor_ui_task.done():
            logger.debug(
                "Cancelling monitor_ui task for session", session_id=self.session_id
            )
            self.monitor_ui_task.cancel()
            self.monitor_ui_task = None

    async def cleanup(self) -> None:
        """Clean up session resources with timeout and verification."""
        logger.debug("Starting cleanup for session", session_id=self.session_id)

        try:
            # Set status to terminating (Phase 2 enhancement)
            self.status = SessionStatus.TERMINATING

            # Cancel monitor_ui task first to prevent new WebSocket operations
            self.cancel_monitor_ui_task()

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
            logger.debug(
                "Cleaning up resources for session", session_id=self.session_id
            )
            cleanup_report = await self.resources.cleanup()

            # Detach OTEL baggage if attached
            try:
                if self._otel_baggage_token is not None:
                    detach_session_baggage(self._otel_baggage_token)
                    self._otel_baggage_token = None
            except Exception:
                pass

            # Session root span removed (Phase 1 OTEL fix)
            # No cleanup needed - session context propagated via baggage only
            # _otel_session_root is always None now

            # Log cleanup report
            if cleanup_report.get("errors"):
                logger.warning(
                    "Session cleanup completed with errors",
                    session_id=self.session_id,
                    cleanup_report=cleanup_report,
                )
                self.status = SessionStatus.ERROR
            else:
                logger.info(
                    "Session cleaned up successfully",
                    session_id=self.session_id,
                    cleanup_report=cleanup_report,
                )
                self.status = SessionStatus.TERMINATED

        except Exception as e:
            logger.error(
                "Error during session cleanup", session_id=self.session_id, error=str(e)
            )
            # Ensure status is set even if cleanup fails
            self.status = SessionStatus.ERROR

    async def monitor_ui(self) -> AsyncGenerator[RunRequest, None]:
        """Monitor the UI for incoming messages."""
        logger.debug(
            "[MONITOR_UI] Starting monitor_ui for session", session_id=self.session_id
        )
        while True:
            await asyncio.sleep(0.1)

            if not self.websocket:
                continue

            try:
                data = await self.websocket.receive_json()
                logger.debug(
                    "[MONITOR_UI] Received message from WebSocket for session",
                    message_type=data.get("type", "unknown"),
                    session_id=self.session_id,
                )
                self.update_activity()  # Update activity timestamp on message

                message = await MessageService.process_message_from_ui(data)
                if not message:
                    logger.debug(
                        "[MONITOR_UI] No message returned from process_message_from_ui",
                        data=data,
                    )
                    continue

                if isinstance(message, RunRequest):
                    # Generate a request to run the flow with the new parameters
                    message.callback_to_ui = self.send_message_to_ui
                    message.session_id = self.session_id
                    logger.info(
                        "Yielding RunRequest for flow in session",
                        flow=message.flow,
                        session_id=self.session_id,
                    )

                    yield message
                elif not self.callback_to_groupchat:
                    # Group chat has not started yet
                    logger.debug(
                        "Group chat not yet started for session",
                        session_id=self.session_id,
                    )
                    continue
                else:
                    await self.callback_to_groupchat(message)
            except WebSocketDisconnect:
                logger.debug(
                    "WebSocket disconnected for session", session_id=self.session_id
                )
                self.websocket = None
                break
            except Exception as e:
                logger.error(
                    "Error receiving/processing client message",
                    session_id=self.session_id,
                    error=str(e),
                )
                self.websocket = None
                break
                # raise FatalError(f"Error receiving/processing client message for {self.session_id}: {e}")

    async def send_message_to_ui(
        self,
        message: ExecutionTrace
        | SystemPromptMessage
        | Record
        | FlowEvent
        | FlowMessage,
    ) -> None:
        """Send a message to a WebSocket connection.

        Args:
            message: The message to send

        """
        message_type = type(message).__name__
        formatted_message = MessageService.format_message_for_client(message)
        if not formatted_message:
            logger.debug(
                "Unhandled message type, not forwarding to UI",
                message_type=message_type,
            )
            return

        # Persist message to session storage
        try:
            storage_service = SessionStorageService()
            if storage_service.should_persist_message(formatted_message):
                storage_service.save_message(self.session_id, formatted_message)
        except Exception as e:
            logger.warning(
                "Failed to persist message for session",
                session_id=self.session_id,
                error=str(e),
            )
            # Continue even if persistence fails

        if self.websocket is None:
            return

        try:
            message_type = formatted_message.type
            message_data_to_send = scrub_serializable(
                formatted_message.model_dump(exclude_unset=True, exclude_none=True)
            )

            # Consolidate debug info into a single log entry
            logger.debug(
                "Flowrunner sending message to ui",
                message_type=message_type,
                session_id=self.session_id,
                ws_state=self.websocket.client_state,
                message_data=json.dumps(message_data_to_send)[:100],
            )

            async def _send_with_retry_internal():
                if not self.websocket:
                    # Raise an error to be caught by tenacity or the outer try/except
                    raise RuntimeError(
                        f"WebSocket is None for session {self.session_id} during send attempt."
                    )

                await self.websocket.send_json(message_data_to_send)

            await _send_with_retry_internal()

        except Exception as e:
            # Attempt to send an error message back to the client if the websocket is still viable
            websocket_state = (
                self.websocket.client_state if self.websocket else "websocket is None"
            )
            if self.websocket:
                try:
                    error_event = ErrorEvent(
                        source="websocket_manager",
                        content=f"Failed to send message to client: {e!s}",
                    )
                    error_message_data = {
                        "content": error_event.model_dump(),
                        "type": "system_message",
                    }
                    await self.websocket.send_json(error_message_data)
                except Exception as err2:
                    logger.warning(
                        "Failed to send message and error notification to UI",
                        session_id=self.session_id,
                        error=e,
                        error2=err2,
                        message_type=message_type,
                        websocket_state=websocket_state,
                    )
                    return
            logger.warning(
                "Cannot send message to UI for session",
                session_id=self.session_id,
                websocket_state=websocket_state,
                message_type=message_type,
                error=str(e),
            )


class OrchestratorFactory:
    """Factory for creating and managing orchestrator instances with proper lifecycle."""

    @staticmethod
    def create_orchestrator(
        flow_config: OrchestratorProtocol, flow_name: str
    ) -> Orchestrator:
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

            logger.debug(
                "Created fresh orchestrator for flow",
                flow_name=flow_name,
                orchestrator_class=orchestrator_cls.__name__,
            )
            return orchestrator

        except Exception as e:
            raise ValueError(
                f"Failed to create orchestrator for flow '{flow_name}': {e}"
            ) from e

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
                logger.debug(
                    "Orchestrator cleanup completed",
                    orchestrator_class=orchestrator.__class__.__name__,
                )
            except Exception as e:
                logger.warning("Error during orchestrator cleanup", error=str(e))


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
        self.active_connections: dict[
            str, set[Any]
        ] = {}  # Multiple connections per session
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

    async def get_or_create_session(
        self, session_id: str, websocket: Any = None
    ) -> FlowRunContext:
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
                    # Ensure OTEL baggage is attached for this session context
                    try:
                        if getattr(session, "_otel_baggage_token", None) is None:
                            session._otel_baggage_token = attach_session_baggage(
                                session_id
                            )
                    except Exception:
                        pass
                    logger.debug(
                        "Added WebSocket to existing session", session_id=session_id
                    )
                return session

            # Create new session with INITIALIZING status
            session = FlowRunContext(
                session_id=session_id,
                websocket=websocket,
                session_timeout=self.session_timeout,
                status=SessionStatus.INITIALIZING,
            )

            # Initialize session infrastructure
            self.sessions[session_id] = session
            self.session_locks[session_id] = asyncio.Lock()
            self.session_resources[session_id] = SessionResources()
            self.active_connections[session_id] = set()

            if websocket:
                self.active_connections[session_id].add(websocket)
                session.add_websocket(websocket)
            # Attach OTEL baggage for this new session
            try:
                session._otel_baggage_token = attach_session_baggage(session_id)
            except Exception:
                pass

            # Session root span removed (Phase 1 OTEL fix)
            # Session context now propagated via OTEL baggage only
            # Each flow run creates its own independent root trace
            session._otel_session_root = None

            logger.info(
                "Created new session with INITIALIZING status", session_id=session_id
            )

            # Transition to ACTIVE after initialization
            await self._transition_session_status(session_id, SessionStatus.ACTIVE)

            return session

    async def _transition_session_status(
        self, session_id: str, new_status: SessionStatus
    ) -> bool:
        """Safely transition a session to a new status.

        Args:
            session_id: The session to transition
            new_status: The new status to transition to

        Returns:
            True if transition was successful, False otherwise

        """
        if session_id not in self.sessions:
            logger.warning(
                "Attempted to transition status for non-existent session",
                session_id=session_id,
            )
            return False

        session = self.sessions[session_id]
        old_status = session.status

        # Validate transition (enhanced validation for Phase 2)
        valid_transitions = {
            SessionStatus.INITIALIZING: [SessionStatus.ACTIVE, SessionStatus.ERROR],
            SessionStatus.ACTIVE: [
                SessionStatus.TERMINATING,
                SessionStatus.RECONNECTING,
                SessionStatus.COMPLETED,
                SessionStatus.ERROR,
                SessionStatus.EXPIRED,
            ],
            SessionStatus.RECONNECTING: [
                SessionStatus.ACTIVE,
                SessionStatus.TERMINATING,
                SessionStatus.ERROR,
            ],
            SessionStatus.TERMINATING: [SessionStatus.TERMINATED, SessionStatus.ERROR],
            SessionStatus.COMPLETED: [SessionStatus.TERMINATED],
            SessionStatus.TERMINATED: [],  # Terminal state - no transitions allowed
            SessionStatus.ERROR: [SessionStatus.TERMINATING, SessionStatus.TERMINATED],
            SessionStatus.EXPIRED: [
                SessionStatus.TERMINATING,
                SessionStatus.TERMINATED,
            ],
            SessionStatus.PAUSED: [
                SessionStatus.ACTIVE,
                SessionStatus.TERMINATING,
                SessionStatus.ERROR,
            ],  # Legacy support
            SessionStatus.FAILED: [
                SessionStatus.TERMINATING,
                SessionStatus.TERMINATED,
            ],  # Legacy support
        }

        if (
            old_status in valid_transitions
            and new_status not in valid_transitions[old_status]
        ):
            logger.warning(
                "Invalid status transition for session",
                session_id=session_id,
                old_status=old_status.value,
                new_status=new_status.value,
            )
            return False

        session.status = new_status
        logger.debug(
            "Session status transition",
            session_id=session_id,
            old_status=old_status.value,
            new_status=new_status.value,
        )
        # Update session-root span attribute to reflect status change
        try:
            if session._otel_session_root is not None:
                span, _ = session._otel_session_root
                if span is not None:
                    span.set_attribute("buttermilk.session.status", new_status.value)
        except Exception:
            pass

        # Update session storage flow status
        try:
            storage_service = SessionStorageService()

            # Terminal states - finalize with archival
            if new_status in {SessionStatus.COMPLETED, SessionStatus.TERMINATED}:
                storage_service.finalize_session(session_id, "completed")
            elif new_status in {
                SessionStatus.ERROR,
                SessionStatus.FAILED,
                SessionStatus.EXPIRED,
            }:
                storage_service.finalize_session(session_id, "failed")
            else:
                # Non-terminal states - just update status
                storage_service.update_flow_status(session_id, "running")

        except Exception as e:
            logger.warning(
                "Failed to update session storage", session_id=session_id, error=str(e)
            )

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
                    logger.warning(
                        "Error in custom shutdown handler for session",
                        session_id=session_id,
                        error=str(e),
                    )
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
            logger.info(
                "Session removed and cleaned up (TERMINATED)", session_id=session_id
            )
            return True

    async def register_shutdown_handler(
        self, session_id: str, handler: Callable
    ) -> None:
        """Register a custom shutdown handler for a session.

        Args:
            session_id: The session to register the handler for
            handler: An async callable that performs custom cleanup

        """
        self.shutdown_handlers[session_id] = handler
        logger.debug("Registered shutdown handler for session", session_id=session_id)

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
            logger.info("Session has expired", session_id=session_id)
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
            logger.warning(
                "Attempted to handle disconnect for non-existent session",
                session_id=session_id,
            )
            return False

        session = self.sessions[session_id]

        # Only allow reconnection for ACTIVE sessions
        if session.status != SessionStatus.ACTIVE:
            logger.debug(
                "Session in status - cleaning up instead of allowing reconnection",
                session_id=session_id,
                status=session.status.value,
            )
            await self.cleanup_session(session_id)
            return False

        # Transition to RECONNECTING status
        success = await self._transition_session_status(
            session_id, SessionStatus.RECONNECTING
        )
        if success:
            # Clear the websocket but keep the session alive
            session.websocket = None
            # Clear active connections for this session
            if session_id in self.active_connections:
                self.active_connections[session_id].clear()

            logger.debug(
                "Session transitioned to RECONNECTING - client can reconnect within timeout",
                session_id=session_id,
                timeout_seconds=session.session_timeout,
            )
            return True
        # Transition failed, clean up the session
        await self.cleanup_session(session_id)
        return False

    async def reconnect_session(
        self, session_id: str, websocket: Any
    ) -> FlowRunContext | None:
        """Reconnect a client to an existing session in RECONNECTING status.

        Args:
            session_id: The session to reconnect to
            websocket: The new WebSocket connection

        Returns:
            The session if reconnection was successful, None otherwise

        """
        if session_id not in self.sessions:
            logger.warning(
                "Attempted to reconnect to non-existent session", session_id=session_id
            )
            return None

        session = self.sessions[session_id]

        # Only allow reconnection to sessions in RECONNECTING status
        if session.status != SessionStatus.RECONNECTING:
            logger.warning(
                "Attempted to reconnect to session in status",
                session_id=session_id,
                status=session.status.value,
            )
            return None

        # Check if session has expired
        if session.is_expired():
            logger.info(
                "Session has expired - cleaning up instead of reconnecting",
                session_id=session_id,
            )
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
        success = await self._transition_session_status(
            session_id, SessionStatus.ACTIVE
        )
        if success:
            logger.debug("Successfully reconnected session", session_id=session_id)
            return session
        logger.error(
            "Failed to transition session back to ACTIVE after reconnection",
            session_id=session_id,
        )
        return None

    async def _periodic_cleanup(self) -> None:  # noqa: PLR0912
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
                        time_since_completion = (
                            datetime.now(UTC) - session.last_activity
                        ).total_seconds()
                        if time_since_completion > 300:  # 5 minutes grace period
                            cleanup_candidates.append(
                                (session_id, "completed_gracetime")
                            )
                    elif session.status == SessionStatus.RECONNECTING:
                        # Clean up RECONNECTING sessions that have exceeded timeout
                        if session.is_expired():
                            cleanup_candidates.append((session_id, "reconnect_timeout"))
                    elif len(self.active_connections.get(session_id, set())) == 0:
                        # No active connections for extended period
                        time_since_activity = (
                            datetime.now(UTC) - session.last_activity
                        ).total_seconds()
                        if time_since_activity > 1800:  # 30 minutes without connections
                            cleanup_candidates.append((session_id, "no_connections"))

                # Perform cleanup
                for session_id, reason in cleanup_candidates:
                    logger.info(
                        "Cleaning up session", session_id=session_id, reason=reason
                    )
                    await self.cleanup_session(session_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in session cleanup task", error=str(e))


class FlowRunner(BaseModel):
    """Centralized service for running flows across different entry points.

    Handles orchestrator instantiation and execution in a consistent way, regardless
    of whether the flow is started from CLI, API, Slackbot, or Pub/Sub.

    The FlowRunner can accept a BM instance for session-scoped operations,
    or fall back to the global singleton for backward compatibility.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    flows: dict[str, OrchestratorProtocol]

    # Session-scoped BM instance (optional)
    bm: Any | None = Field(
        default=None,
        description="Optional session-scoped BM instance. If None, falls back to global singleton.",
    )

    tasks: list = Field(default=[])
    mode: str = Field(default="api")
    ui: str = Field(default="console")
    human_in_loop: bool = False
    sessions: dict[str, FlowRunContext] = Field(
        default_factory=dict
    )  # Dictionary of active sessions (DEPRECATED)

    # New session management
    session_manager: SessionManager = Field(default_factory=lambda: SessionManager())
    _session_manager_started: bool = False

    async def _ensure_session_manager_started(self) -> None:
        """Ensure the session manager is started."""
        if not self._session_manager_started:
            await self.session_manager.start()
            self._session_manager_started = True

    def set_session_bm(self, bm: Any) -> None:
        """Set a session-scoped BM instance for this FlowRunner.

        This enables session-level observability isolation by providing each flow execution
        with its own BM instance containing unique session context (session_id, job, platform).
        When set, all orchestrators and agents created by this FlowRunner will automatically
        receive the session-scoped BM instead of the global singleton.

        Args:
            bm: Session-scoped BM instance containing unique session context for observability.
                Must have session_info.session_id for proper isolation.

        Example:
            >>> from buttermilk._core.config_bootstrap import create_configuration_bootstrapper
            >>> import asyncio
            >>> bootstrapper = create_configuration_bootstrapper()
            >>> session_bm = asyncio.run(bootstrapper.bootstrap_session_context(
            ...     name="api_session", job="analysis", platform="local"
            ... ))
            >>> flow_runner.set_session_bm(session_bm)
            >>> # All flows will now use session-scoped observability
        """
        self.bm = bm
        logger.debug("Set session-scoped BM", session_id=bm.session_info.session_id)

    def get_effective_bm(self) -> Any:
        """Get the effective BM instance (session-scoped if available, otherwise global singleton).

        This method implements the dependency injection pattern for BM access, providing
        session-scoped observability when available while maintaining backward compatibility
        with the global singleton pattern.

        Returns:
            BM instance to use for operations. Session-scoped if set via set_session_bm(),
            otherwise the global singleton from get_bm().

        Raises:
            RuntimeError: If no session-scoped BM is set and global singleton is not initialized.

        Example:
            >>> bm = flow_runner.get_effective_bm()
            >>> # Gets session-scoped BM if available, otherwise global singleton
            >>> session_id = bm.session_info.session_id  # Unique per session
        """
        if self.bm is not None:
            return self.bm
        else:
            from buttermilk._core.dmrc import get_bm

            return get_bm()

    async def get_websocket_session_async(
        self, session_id: str, websocket: Any | None = None
    ) -> FlowRunContext | None:
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
                logger.debug(
                    "Attempting to reconnect to session", session_id=session_id
                )
                reconnected_session = await self.session_manager.reconnect_session(
                    session_id, websocket
                )
                if reconnected_session:
                    return reconnected_session
                # Reconnection failed, fall through to create new session
                logger.warning(
                    "Failed to reconnect to session, creating new session",
                    session_id=session_id,
                )
            elif existing_session.status == SessionStatus.TERMINATED:
                # Session has been terminated, don't allow new connections
                logger.info(
                    "WebSocket connection attempt to terminated session, rejecting",
                    session_id=session_id,
                )
                return None
            elif existing_session.status in [
                SessionStatus.ACTIVE,
                SessionStatus.INITIALIZING,
            ]:
                # Session is already active, replace the websocket connection
                if websocket:
                    # Cancel existing monitor_ui task to prevent WebSocket conflicts
                    existing_session.cancel_monitor_ui_task()

                    # Close existing websocket if any
                    if (
                        existing_session.websocket
                        and existing_session.websocket != websocket
                    ):
                        try:
                            logger.debug(
                                "Closing previous WebSocket connection for session",
                                session_id=session_id,
                            )
                            await existing_session.websocket.close()
                        except Exception as e:
                            logger.warning(
                                "Error closing previous WebSocket for session",
                                session_id=session_id,
                                error=str(e),
                            )

                    # Replace with new websocket
                    existing_session.websocket = websocket
                    existing_session.add_websocket(websocket)
                    logger.debug(
                        "Replaced WebSocket connection for active session",
                        session_id=session_id,
                    )
                return existing_session

        # Create new session if no websocket provided or reconnection failed
        if not websocket:
            return None

        session = await self.session_manager.get_or_create_session(
            session_id, websocket
        )
        return session

    async def cleanup(self) -> None:
        """Clean up the FlowRunner and all its sessions."""
        if self._session_manager_started:
            await self.session_manager.stop()
            self._session_manager_started = False

    async def reload_configurations(self) -> dict[str, Any]:
        """Reload flow configurations from the mounted GCS config directory.

        This method re-reads the configuration files and updates the flows without
        disrupting active sessions. It uses Hydra to reload configurations from
        the current config directory (which may be GCS-mounted).

        Returns:
            A dictionary containing reload status and details:
            - success: Whether the reload was successful
            - flows_loaded: List of flow names that were loaded
            - flows_updated: List of flow names that were updated
            - errors: List of any errors encountered
            - timestamp: When the reload occurred
        """
        import traceback
        from datetime import UTC, datetime
        from pathlib import Path

        import hydra
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        old_flows = None

        result = {
            "success": False,
            "flows_loaded": [],
            "flows_updated": [],
            "flows_removed": [],
            "errors": [],
            "timestamp": datetime.now(UTC).isoformat(),
            "config_source": "unknown",
        }

        try:
            logger.info("Starting configuration reload...")

            # Store current flows for comparison
            current_flows = set(self.flows.keys())

            # Get the current config directory
            config_dir = Path("/src/buttermilk/buttermilk/conf").resolve()
            result["config_source"] = str(config_dir)

            # Check if config directory exists and has required files
            if not config_dir.exists():
                raise ValueError(f"Configuration directory not found: {config_dir}")

            config_yaml = config_dir / "config.yaml"
            if not config_yaml.exists():
                raise ValueError(f"Main config file not found: {config_yaml}")

            logger.info("Reloading configuration from directory", config_dir=config_dir)

            # Clear Hydra's global state to ensure fresh load
            hydra.core.global_hydra.GlobalHydra.instance().clear()

            # Initialize Hydra with the config directory
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
                # Compose the configuration using the same pattern as CLI
                conf = compose(config_name="config")
                OmegaConf.resolve(conf)

                # Extract flows from the reloaded configuration
                if hasattr(conf, "run") and hasattr(conf.run, "flows"):
                    new_flows = conf.run.flows
                    logger.info(
                        "Found flows in reloaded config", flow_count=len(new_flows)
                    )

                    # Update flows dictionary
                    old_flows = self.flows.copy()
                    self.flows = new_flows

                    # Determine what changed
                    new_flow_names = set(new_flows.keys())

                    result["flows_loaded"] = list(new_flow_names)
                    result["flows_updated"] = list(
                        current_flows.intersection(new_flow_names)
                    )
                    result["flows_removed"] = list(current_flows - new_flow_names)

                    # Log the changes
                    if result["flows_updated"]:
                        logger.info(
                            "Updated flows", flows_updated=result["flows_updated"]
                        )
                    if new_flow_names - current_flows:
                        logger.info(
                            "New flows", new_flows=list(new_flow_names - current_flows)
                        )
                    if result["flows_removed"]:
                        logger.info(
                            "Removed flows", flows_removed=result["flows_removed"]
                        )

                    result["success"] = True
                    logger.info("Configuration reload completed successfully")

                else:
                    raise ValueError("No flows found in reloaded configuration")

        except Exception as e:
            error_msg = f"Configuration reload failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)
            result["errors"].append(traceback.format_exc())

            # Don't leave flows in broken state - keep old flows on error
            if old_flows is not None:
                self.flows = old_flows
                logger.info("Restored previous flow configuration due to reload error")

        finally:
            # Clean up Hydra state
            try:
                hydra.core.global_hydra.GlobalHydra.instance().clear()
            except Exception as cleanup_error:
                logger.warning(
                    "Error cleaning up Hydra state", error=str(cleanup_error)
                )

        return result

    def _save_config_snapshot(self, run_request: "RunRequest") -> None:
        """Save a snapshot of the current configuration for reproducibility.

        Saves one config file per session (named by session_id) in the same location
        as session message logs. File descriptors are properly closed after writing.

        Args:
            run_request: The run request containing flow and session information
        """
        try:
            import json
            from pathlib import Path

            from omegaconf import OmegaConf

            # Get session ID, defaulting to "default" if not available
            session_id = getattr(run_request, "session_id", "default") or "default"

            # Use session_id as the base filename (same pattern as message logs)
            # Store in /tmp/runs/{session_id}/ alongside message logs
            session_dir = Path(f"/tmp/runs/{session_id}")
            session_dir.mkdir(parents=True, exist_ok=True)

            # Save flow configuration with session_id as base name
            if run_request.flow in self.flows:
                flow_config = self.flows[run_request.flow]

                # Convert OmegaConf to serializable dict
                if hasattr(flow_config, "_content"):
                    # OmegaConf object
                    config_dict = OmegaConf.to_container(flow_config, resolve=True)
                else:
                    # Regular dict or other object
                    config_dict = (
                        dict(flow_config)
                        if hasattr(flow_config, "__dict__")
                        else str(flow_config)
                    )

                # Config file named by session_id (e.g., /tmp/runs/abc123/abc123_config.json)
                # This overwrites on each run, keeping only one config file per session
                config_file = session_dir / f"{session_id}_config.json"

                # Use 'with' statement to ensure file descriptor is closed
                with open(config_file, "w") as f:
                    json.dump(
                        {
                            "flow_name": run_request.flow,
                            "timestamp": datetime.now(UTC).isoformat(),
                            "session_id": session_id,
                            "job_id": getattr(run_request, "job_id", None),
                            "flow_config": config_dict,
                            "run_parameters": getattr(run_request, "parameters", {}),
                            "run_inputs": getattr(run_request, "inputs", {}),
                            "flows_available": list(self.flows.keys()),
                            "total_flows": len(self.flows),
                        },
                        f,
                        indent=2,
                        default=str,
                    )
                # File is automatically closed here when exiting 'with' block

                logger.debug(
                    "Saved config snapshot for session",
                    session_id=session_id,
                    config_file=str(config_file),
                )

        except Exception as e:
            # Don't fail the flow execution if config snapshot fails
            logger.warning(
                "Failed to save config snapshot",
                session_id=getattr(run_request, "session_id", "unknown"),
                error=str(e),
            )

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
            raise ValueError(
                f"Flow '{flow_name}' not found. Available flows: {list(self.flows.keys())}"
            )

        flow_config = self.flows[flow_name]
        orchestrator = OrchestratorFactory.create_orchestrator(flow_config, flow_name)

        # Inject session-scoped BM for observability isolation
        # This ensures each flow execution gets its own observability context
        # (session_id, job, platform) instead of sharing global singleton state
        if self.bm is not None:
            orchestrator.set_bm(self.bm)
            logger.debug(
                "Injected session-scoped BM into orchestrator for flow",
                flow_name=flow_name,
                session_id=self.bm.session_info.session_id,
            )
        else:
            logger.debug(
                "Using global singleton BM for orchestrator (legacy mode)",
                flow_name=flow_name,
            )

        return orchestrator

    async def _cleanup_flow_context(self, context: FlowRunContext) -> None:
        """Clean up resources associated with a flow run.

        Args:
            context: The flow run context to clean up

        """
        # Use the enhanced cleanup from FlowRunContext
        await context.cleanup()

    async def run_flow(
        self, run_request: RunRequest, wait_for_completion: bool = False, **kwargs
    ) -> None:  # noqa: PLR0912
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
        # Use injected BM
        bm = self.bm

        # Ensure BM is fully initialized before running flow
        if hasattr(bm, "ensure_initialized"):
            await bm.ensure_initialized()
            logger.debug("BM initialization verified before flow execution")

        # Phase 1 OTEL fix: Validate session ID consistency
        # NOTE: We do NOT validate session_id match between BM and request
        # Each job creates its own session and can be run by any worker.
        # The BM instance's session is for the worker process, not the job.

        # Initialize metrics tracking
        start_time = time.time()
        success = False

        try:
            from buttermilk.monitoring import get_metrics_collector

            metrics_collector = get_metrics_collector()
        except Exception as e:
            logger.debug("Failed to initialize metrics collector", error=str(e))
            metrics_collector = None

        # Wrap the flow execution in a ROOT tracing span (detached from any parent)
        # This ensures each flow execution creates an independent trace, preventing
        # traces from nesting when multiple jobs run in the same worker process.
        # Session baggage is already attached during BM initialization.
        with start_root_span(
            name="buttermilk.flow.run",
            attributes={
                "buttermilk.session.id": getattr(run_request, "session_id", None),
                "buttermilk.flow.name": getattr(run_request, "flow", None),
                "buttermilk.job.id": getattr(run_request, "job_id", None),
                "buttermilk.source": ", ".join(run_request.source)
                if getattr(run_request, "source", None)
                else "direct",
                "buttermilk.mode": self.mode,
            },
            kind="internal",
        ) as span:
            if span is not None:
                span.add_event("flow_start")

            # Ensure session manager is started
            await self._ensure_session_manager_started()

            # Create a fresh orchestrator instance
            fresh_orchestrator = self._create_fresh_orchestrator(run_request.flow)

            # set a high max callback duration when dealing with LLMs
            asyncio.get_event_loop().slow_callback_duration = 120

            # Get or create session using the session manager
            _session = await self.session_manager.get_or_create_session(
                run_request.session_id
            )

            set_logging_context(run_request.session_id)
            _session.flow_name = run_request.flow
            _session.orchestrator = fresh_orchestrator
            _session.callback_to_groupchat = fresh_orchestrator.make_publish_callback()
            _session.update_activity()  # Update activity timestamp

            # Save configuration snapshot for reproducibility
            self._save_config_snapshot(run_request)

        # Set the callback_to_ui for the run_request, which will be used by the orchestrator
        run_request.callback_to_ui = _session.send_message_to_ui
        logger.debug(
            "[FlowRunner.run_flow] Callback configured for session",
            session_id=_session.session_id,
        )

        # Create the task and register it with the session
        _session.flow_task = asyncio.create_task(
            fresh_orchestrator.run(request=run_request)
        )  # type: ignore
        _session.add_task(_session.flow_task)

        # ======== MAJOR EVENT: FLOW STARTING ========
        # Log detailed information about flow start
        logger.info(
            f"🚀 FLOW STARTING: '{run_request.flow}' (ID: {run_request.job_id}) | "
            f"Source: {', '.join(run_request.source) if run_request.source else 'direct'} | "
            f"New flow instance created",
        )

        # Update session storage to mark flow as running
        try:
            storage_service = SessionStorageService()
            storage_service.update_flow_status(run_request.session_id, "running")
            logger.debug(
                "Updated flow status to 'running' for session",
                session_id=run_request.session_id,
            )

            # Save flow parameters for demo mode functionality
            parameters = {
                "flow": run_request.flow,
            }

            # Extract record_id and dataset from inputs if available
            if hasattr(run_request, "inputs") and run_request.inputs:
                if "record_id" in run_request.inputs:
                    parameters["record_id"] = run_request.inputs["record_id"]
                if "dataset" in run_request.inputs:
                    parameters["dataset"] = run_request.inputs["dataset"]

            # Extract criteria from parameters if available
            if hasattr(run_request, "parameters") and run_request.parameters:
                if "criteria" in run_request.parameters:
                    parameters["criteria"] = run_request.parameters["criteria"]

            storage_service.save_parameters(run_request.session_id, parameters)
            logger.debug(
                "Saved flow parameters for session",
                parameters=parameters,
                session_id=run_request.session_id,
            )

        except Exception as e:
            logger.warning(
                "Failed to update session storage flow status",
                session_id=run_request.session_id,
                error=str(e),
            )

        try:
            if wait_for_completion:
                # Wait for the task
                await _session.flow_task
                await self.session_manager._transition_session_status(
                    run_request.session_id, SessionStatus.COMPLETED
                )
                success = True
                return
        except Exception as e:
            await self.session_manager._transition_session_status(
                run_request.session_id, SessionStatus.ERROR
            )
            logger.error("Error running flow", flow=run_request.flow, error=str(e))
            raise
        finally:
            # Record flow execution metrics
            if metrics_collector:
                execution_time = time.time() - start_time
                try:
                    metrics_collector.record_flow_execution(
                        flow_name=run_request.flow,
                        execution_time=execution_time,
                        success=success,
                    )
                except Exception as e:
                    logger.debug(
                        "Failed to record flow execution metrics", error=str(e)
                    )

            if wait_for_completion:
                # Clean up after completion if we were waiting
                await self.session_manager.cleanup_session(run_request.session_id)
        return

    async def create_batch(
        self,
        flow_name,
        storage_config: dict | str | None = None,
        max_records: int | None = None,
    ) -> list[RunRequest]:  # noqa: PLR0912
        """Create a new batch job from storage source.

        Args:
            flow_name: Name of the flow to execute
            storage_config: Storage configuration. Can be:
                - dict: Direct storage configuration (pipeline pattern)
                - str: Key into flow.storage dict (backward compat with dataset_key)
                - None: Auto-discover from flow.storage (uses 'initial' or first available)
            max_records: Maximum number of records to process

        Returns:
            List of created RunRequest objects

        Raises:
            ValueError: If the flow doesn't exist or storage cannot be resolved

        """
        # Resolve storage configuration
        flow = self.flows[flow_name]

        if storage_config is None:
            # Auto-discover: prefer 'initial' key, fallback to first available
            if hasattr(flow, "storage") and flow.storage:
                if "initial" in flow.storage:
                    storage_cfg = flow.storage["initial"]
                    logger.debug(
                        "Auto-discovered storage using 'initial' key",
                        flow_name=flow_name,
                    )
                else:
                    storage_cfg = next(iter(flow.storage.values()))
                    logger.debug(
                        "Auto-discovered storage using first available",
                        flow_name=flow_name,
                    )
            else:
                raise ValueError(
                    f"Flow '{flow_name}' has no storage configuration and none was provided"
                )
        elif isinstance(storage_config, str):
            # Legacy dataset_key behavior - lookup in flow.storage
            if not hasattr(flow, "storage") or storage_config not in flow.storage:
                available = (
                    list(flow.storage.keys()) if hasattr(flow, "storage") else []
                )
                raise ValueError(
                    f"Storage key '{storage_config}' not found in flow '{flow_name}'. Available: {available}"
                )
            storage_cfg = flow.storage[storage_config]
            logger.debug(
                "Using storage from key",
                storage_key=storage_config,
                flow_name=flow_name,
            )
        else:
            # Direct storage configuration dict (pipeline pattern)
            storage_cfg = storage_config
            logger.debug("Using direct storage configuration", flow_name=flow_name)

        # Create storage instance using BM
        from buttermilk._core.dmrc import get_bm

        bm = get_bm()
        storage = bm.get_storage(storage_cfg)

        # Stream records from storage (don't load all into memory)
        records = list(storage)  # Storage.__iter__ yields BaseRecord objects
        logger.info(
            "Extracted records from storage",
            record_count=len(records),
            flow_name=flow_name,
        )

        # Create multiple iterations by multiplying the parameters
        iteration_values = expand_dict(flow.parameters) or [{}]
        logger.debug(
            "Expanded parameters for batch into variants",
            parameter_count=len(flow.parameters),
            variant_count=len(iteration_values),
        )

        #
        # Shuffle records
        random.shuffle(records)

        batch_id = str(shortuuid.uuid())

        # Create run requests for each record and parameter combination
        job_definitions = []

        # Apply iteration values
        for iteration_params in iteration_values:
            for i, record in enumerate(records):
                data = {"record_id": record.record_id}
                job = RunRequest(
                    batch_id=batch_id,
                    flow=flow_name,
                    # session_id gets auto-generated UUID - each job is independent
                    parameters=iteration_params,
                    inputs=data,
                    callback_to_ui=None,
                )
                job_definitions.append(job)
                logger.debug(
                    "Batch job created",
                    flow=flow_name,
                    record_id=record.record_id,
                    job_id=job.job_id,
                )
                # Apply max_records limit if specified
                if max_records is not None and max_records > 0 and i >= max_records:
                    break

        if max_records is not None and max_records > 0:
            logger.info(
                "Limited record IDs, returning iterations for jobs",
                max_records=max_records,
                iteration_count=len(iteration_values),
                job_count=len(job_definitions),
            )
        else:
            logger.info(
                "Returning iterations for jobs",
                iteration_count=len(iteration_values),
                job_count=len(job_definitions),
            )

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

    async def run_batch_job(
        self,
        callback_to_ui: Callable,
        max_jobs: int = 1,
        wait_for_completion: bool = True,
        show_progress: bool = True,
    ) -> ProcessingSummary:
        """Pull and run jobs from the queue, ensuring fresh state for each job.

        Args:
            max_jobs: Maximum number of jobs to process in this batch run
            callback_to_ui: Callback function for UI updates
            wait_for_completion: Whether to wait for each job to complete
            show_progress: Whether to display a progress bar (default: True)

        Returns:
            ProcessingSummary: Statistics about the batch processing

        Raises:
            FatalError: If no run requests are found in the queue
            Exception: If there's an error running a job

        """
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        summary = ProcessingSummary()

        try:
            worker = JobQueueClient(
                max_concurrent_jobs=1,  # Process one job at a time to maintain isolation
            )

            jobs_processed = 0

            # Set up progress bar
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                disable=not show_progress,
            )

            with progress:
                task_id = progress.add_task("Processing batch jobs", total=max_jobs)

                while jobs_processed < max_jobs:
                    # Pull a job from the queue
                    run_request, ack_id = await worker.pull_single_task()
                    if not run_request:
                        if summary.attempted == 0:
                            # Only raise an error if we didn't process any jobs
                            raise FatalError("No run request found in the queue.")
                        # Update progress to show we're done
                        progress.update(task_id, total=summary.attempted)
                        break  # No more jobs to process

                    summary.increment_attempted()
                    run_request.callback_to_ui = callback_to_ui

                    # Update progress description with current job
                    progress.update(
                        task_id,
                        description=f"Processing {run_request.flow} [{run_request.job_id[:8]}...]",
                    )

                    logger.info(
                        "Processing batch job",
                        job_number=jobs_processed + 1,
                        max_jobs=max_jobs,
                        flow=run_request.flow,
                        job_id=run_request.job_id,
                    )
                    try:
                        await self.run_flow(
                            run_request=run_request,
                            wait_for_completion=wait_for_completion,
                        )
                        summary.increment_processed()
                        if wait_for_completion:
                            logger.info(
                                "Successfully completed job", job_id=run_request.job_id
                            )
                            worker.ack_message(
                                ack_id
                            )  # Acknowledge only after successful processing
                        else:
                            logger.info(
                                "Job started in the background",
                                job_id=run_request.job_id,
                            )
                            # Defer ack until the background task completes successfully
                            try:
                                self.schedule_ack_on_completion(
                                    session_id=run_request.session_id,
                                    ack_id=ack_id,
                                    worker=worker,
                                )
                            except Exception as e:
                                logger.warning(
                                    "Failed to schedule ack on completion",
                                    job_id=run_request.job_id,
                                    error=str(e),
                                )
                    except Exception as job_error:
                        summary.increment_failed()
                        logger.error(
                            "Error running job",
                            job_id=run_request.job_id,
                            error=str(job_error),
                        )
                        # Continue processing other jobs even if one fails

                    jobs_processed += 1
                    progress.update(task_id, advance=1)

            logger.info(summary.format_for_console())
            return summary

        except FatalError:
            # Re-raise FatalError to be handled by the caller
            raise
        except Exception as e:
            logger.error("Fatal error during batch processing", error=str(e))
            raise

    def schedule_ack_on_completion(
        self, session_id: str, ack_id: str, worker: JobQueueClient
    ) -> None:
        """Schedule Pub/Sub ack once the flow task completes successfully.

        This ensures messages are only acknowledged after the background flow
        finishes without raising an exception. If the task fails, no ack is sent
        so Pub/Sub can redeliver according to its settings.

        Args:
            session_id: The FlowRunContext session id housing the flow task
            ack_id: The Pub/Sub ack id to acknowledge
            worker: The JobQueueClient used to perform the ack
        """
        session = self.session_manager.sessions.get(session_id)
        if not session:
            logger.warning(
                "Cannot schedule ack: session not found", session_id=session_id
            )
            return

        task = session.flow_task
        if not task or not isinstance(task, asyncio.Task):
            logger.warning(
                "Cannot schedule ack: flow task not available", session_id=session_id
            )
            return

        def _on_done(t: asyncio.Task) -> None:
            try:
                exc = t.exception()
            except asyncio.CancelledError:
                exc = asyncio.CancelledError()

            if exc is None:
                try:
                    worker.ack_message(ack_id)
                    logger.debug(
                        "Acknowledged Pub/Sub message after task completion",
                        session_id=session_id,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to acknowledge Pub/Sub message on completion",
                        session_id=session_id,
                        error=str(e),
                    )
            else:
                logger.error(
                    "Flow task completed with error; not acknowledging message",
                    session_id=session_id,
                    error=str(exc),
                )

        task.add_done_callback(_on_done)
