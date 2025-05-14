import asyncio
import importlib
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import WebSocketDisconnect
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, ConfigDict, Field

from buttermilk import logger
from buttermilk._core import AgentTrace
from buttermilk._core.agent import ErrorEvent
from buttermilk._core.config import SaveInfo
from buttermilk._core.contract import (
    ErrorEvent,
    FlowEvent,
    FlowMessage,
    UIMessage,
)
from buttermilk._core.exceptions import FatalError
from buttermilk._core.orchestrator import Orchestrator, OrchestratorProtocol
from buttermilk._core.types import Record, RunRequest
from buttermilk.api.job_queue import JobQueueClient
from buttermilk.api.services.message_service import MessageService
from buttermilk.bm import BM, logger


class FlowRunContext(BaseModel):
    """Encapsulates all state for a single flow run."""

    flow_name: str = ""
    flow_task: Any | None = None
    orchestrator: Orchestrator | None = None
    status: str = "pending"
    session_id: str
    callback_to_groupchat: Any = None
    messages: list = []
    progress: dict = Field(default_factory=dict)

    websocket: Any = None

    async def monitor_ui(self) -> AsyncGenerator[RunRequest, None]:
        """Monitor the UI for incoming messages."""
        while True:
            await asyncio.sleep(0.1)

            # Check if the WebSocket is connected
            if not self.websocket or self.websocket.client_state != WebSocketState.CONNECTED:
                logger.debug(f"WebSocket not connected for session {self.session_id}")
                await asyncio.sleep(0.1)
                continue

            try:
                data = await self.websocket.receive_json()
                message = await MessageService.process_message_from_ui(data)
                if not message:
                    continue

                if isinstance(message, RunRequest):
                    # Generate a request to run the flow with the new parameters
                    message.callback_to_ui = self.send_message_to_ui
                    message.session_id = self.session_id

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
            except Exception as e:
                logger.error(f"Error receiving/processing client message for {self.session_id}: {e}")
                self.websocket = None
                # raise FatalError(f"Error receiving/processing client message for {self.session_id}: {e}")

    async def send_message_to_ui(self, message: AgentTrace | UIMessage | Record | FlowEvent | FlowMessage) -> None:
        """Send a message to a WebSocket connection.
        
        Args:
            session_id: The session ID
            message: The message to send

        """
        formatted_message = MessageService.format_message_for_client(message)
        if not formatted_message:
            logger.debug(f"Dropping message not handled by client: {message}")
            return
        try:
            message_type = formatted_message.type
            formatted_message = formatted_message.model_dump(mode="json", exclude_unset=True, exclude_none=True)
            message_data = {"content": formatted_message, "type": message_type}

            await self.websocket.send_json(message_data)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            try:
                error_data = ErrorEvent(source="websocket_manager", content=f"Failed to send message: {e!s}")
                message_data = {"content": error_data.model_dump(), "type": "system_message"}
                await self.websocket.send_json(message_data)
            except Exception as e:
                logger.error(f"Failed to send error message: {e!s}")


class FlowRunner(BaseModel):
    """Centralized service for running flows across different entry points.
    
    Handles orchestrator instantiation and execution in a consistent way, regardless
    of whether the flow is started from CLI, API, Slackbot, or Pub/Sub.
    """

    bm: BM
    flows: dict[str, OrchestratorProtocol] = Field(default_factory=dict)  # Flow configurations

    save: SaveInfo
    tasks: list = Field(default=[])
    model_config = ConfigDict(extra="allow")

    sessions: dict[str, FlowRunContext] = Field(default_factory=dict)  # Dictionary of active sessions

    def get_session(self, session_id: str, websocket: Any | None = None) -> FlowRunContext:
        """Get or create a session for the given session ID.
        
        Args:
            session_id: Unique identifier for the session
            websocket: WebSocket connection for this session

        Returns:
            A FlowRunContext object representing the session

        """
        if session_id not in self.sessions:
            self.sessions[session_id] = FlowRunContext(session_id=session_id, websocket=websocket)
        elif self.sessions[session_id].websocket is None and websocket is not None:
            self.sessions[session_id].websocket = websocket
        return self.sessions[session_id]

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
            logger.info(f"Running task from queue: {request.flow} (Job ID: {request.job_id})")
            await self.run_flow(request, wait_for_completion=True)
        else:
            logger.debug("No tasks available in the queue")

    def _create_fresh_orchestrator(self, flow_name: str, session_id: str) -> OrchestratorProtocol:
        """Create a completely fresh orchestrator instance.
        
        Args:
            flow_name: The name of the flow to create an orchestrator for
            session_id: The session ID for the flow

        Returns:
            A new orchestrator instance with fresh state
            
        Raises:
            ValueError: If flow_name doesn't exist in flows

        """
        if flow_name not in self.flows:
            raise ValueError(f"Flow '{flow_name}' not found. Available flows: {list(self.flows.keys())}")

        flow_config = self.flows[flow_name]

        # Extract orchestrator class path
        orchestrator_path = flow_config.orchestrator
        module_name, class_name = orchestrator_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        orchestrator_cls = getattr(module, class_name)

        # Create a fresh config copy to avoid shared state
        config = flow_config.model_dump() if hasattr(flow_config, "model_dump") else dict(flow_config)

        # Create and return a fresh instance
        return orchestrator_cls(**config, session_id=session_id)

    async def _cleanup_flow_context(self, context: FlowRunContext) -> None:
        """Clean up resources associated with a flow run.
        
        Args:
            context: The flow run context to clean up

        """
        # Implement cleanup logic for the orchestrator if available
        cleanup_method = getattr(context.orchestrator, "cleanup", None)
        if cleanup_method is not None and callable(cleanup_method):
            try:
                result = cleanup_method()
                # Handle case where cleanup might be async or not
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Error during orchestrator cleanup: {e}")

        # Set status to completed
        context.status = "completed"

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
        # Create a fresh orchestrator instance
        fresh_orchestrator = self._create_fresh_orchestrator(run_request.flow, session_id=run_request.session_id)

        # Type safety: The orchestrator will be an Orchestrator instance at runtime,
        # even though the flows dict is typed with the more general OrchestratorProtocol
        _session = self.get_session(run_request.session_id)
        _session.flow_name = run_request.flow
        _session.orchestrator = fresh_orchestrator
        _session.callback_to_groupchat = fresh_orchestrator.make_publish_callback()

        # Create the task
        _session.flow_task = asyncio.create_task(fresh_orchestrator.run(request=run_request))  # type: ignore

        # ======== MAJOR EVENT: FLOW STARTING ========
        # Log detailed information about flow start
        logger.info(f"🚀 FLOW STARTING: '{run_request.flow}' (ID: {run_request.job_id}).\n📋 RunRequest: {run_request.model_dump_json(indent=2)}\n⚙️ Source: {', '.join(run_request.source) if run_request.source else 'direct'}\n✅ New flow instance created - all state has been reset")

        try:
            if wait_for_completion:
                # Wait for the task
                result = await _session.flow_task
                _session.status = "completed"
                return
        except Exception as e:
            _session.status = "failed"
            logger.error(f"Error running flow '{run_request.flow}': {e}")
            raise
        finally:
            if wait_for_completion:
                # Clean up after completion if we were waiting
                await self._cleanup_flow_context(_session)
        return
