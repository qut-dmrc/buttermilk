from collections.abc import Callable
from typing import Any

from fastapi import WebSocket

from buttermilk._core import AgentTrace
from buttermilk._core.contract import (
    ErrorEvent,
    FlowEvent,
    FlowMessage,
    ManagerMessage,
    ManagerRequest,
)
from buttermilk._core.types import Record, RunRequest
from buttermilk.api.services.message_service import MessageService
from buttermilk.bm import logger
from buttermilk.runner.flowrunner import FlowRunner

# Session data structure keys
# - messages: list of message objects
# - progress: dict with progress information
# - callback: callable for sending messages back to the flow
# - callback_to_ui: callable for sending messages back to the UI
# - outcomes_version: string timestamp for version tracking


class WebSocketManager:
    """Manages WebSocket connections and message handling"""

    def __init__(self):
        """Initialize the WebSocket manager"""
        self.active_connections: dict[str, WebSocket] = {}
        self.session_data: dict[str, Any] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept a WebSocket connection and store it
        
        Args:
            websocket: The WebSocket connection
            session_id: The session ID

        """
        await websocket.accept()
        self.active_connections[session_id] = websocket
        if session_id not in self.session_data:
            # This is a new session, so we need to initialize the session data
            self.session_data[session_id] = dict(
                messages=[],
                progress={"current_step": 0, "total_steps": 100, "status": "waiting"},
                callback=None,
                callback_to_ui=self.make_callback_to_ui(session_id),
                outcomes_version=None,
            )

    def disconnect(self, session_id: str) -> None:
        """Remove a WebSocket connection
        
        Args:
            session_id: The session ID

        """
        if session_id in self.active_connections:
            del self.active_connections[session_id]

        if session_id in self.session_data:
            del self.session_data[session_id]

    def make_callback_to_ui(self, session_id: str) -> Callable:
        """Create a callback function for the flow to send messages back to the UI
        
        Args:
            session_id: The session ID
            
        Returns:
            Callable: The callback function

        """
        async def callback(message):
            """Handle messages from the flow
            
            Args:
                message: The message from the flow

            """
            await self.send_message(session_id, message)

        return callback

    async def send_message(self, session_id: str, message: AgentTrace | ManagerRequest | Record | FlowEvent | FlowMessage) -> None:
        """Send a message to a WebSocket connection
        
        Args:
            session_id: The session ID
            message: The message to send

        """
        if session_id not in self.active_connections:
            logger.warning(f"Attempted to send message to non-existent session: {session_id}")
            return

        formatted_message = MessageService.format_message_for_client(message)
        if not formatted_message:
            logger.debug(f"Dropping message not handled by client: {message}")
            return
        try:
            message_type = formatted_message.type
            formatted_message = formatted_message.model_dump(mode="json", exclude_unset=True, exclude_none=True)
            message_data = {"content": formatted_message, "type": message_type}

            await self.active_connections[session_id].send_json(message_data)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            try:
                error_data = ErrorEvent(source="websocket_manager", content=f"Failed to send message: {e!s}")
                message_data = {"content": error_data, "type": "system_message"}
                await self.active_connections[session_id].send_json(message_data)
            except Exception as e:
                logger.error("Failed to send error message")

    async def process_message_from_ui(self, session_id: str, message: dict[str, Any], flow_runner: FlowRunner) -> None:
        """Process a message from a WebSocket connection
        
        Args:
            session_id: The session ID
            message: The message to process
            flow_runner: The flow runner instance

        """
        try:
            message_type = message.pop("type", None)

            if message_type == "run_flow":
                run_request = await self.validate_request(session_id, message)
                if run_request:
                    await self.handle_run_flow(session_id, run_request, flow_runner)
            elif message_type == "pull_task":
                from buttermilk.api.job_queue import JobQueueClient
                run_request = await JobQueueClient().pull_single_task()
                if run_request:
                    await self.handle_run_flow(session_id, run_request, flow_runner)
            elif message_type == "manager_response" or message_type == "TaskProcessingComplete" or message_type == "TaskProcessingStarted":
                await self.handle_user_input(session_id, message)
            else:
                await self.send_message(
                    session_id,
                    ErrorEvent(source="websocket_manager", content=f"Unknown message type: {message_type}"),
                )
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.send_message(
                session_id,
                ErrorEvent(source="websocket_manager", content=f"Error processing message: {e!s}"),
            )

    async def validate_request(self, session_id: str, data: dict[str, Any]) -> RunRequest | None:
        """Validate the request data
        
        Args:
            session_id: The session ID
            data: The request data

        """
        if not data.get("flow"):
            await self.send_message(
                session_id,
                ErrorEvent(source="websocket_manager", content="Missing required fields: flow"),
            )
            return None

        # Create run request
        parameters = {}

        # Extract criteria from the original data if present
        if "criteria" in data:
            parameters["criteria"] = data["criteria"]

        run_request = RunRequest(ui_type="web",
            flow=data["flow"],
            record_id=data["record_id"],
            parameters=parameters,
            callback_to_ui=self.make_callback_to_ui(session_id),
            session_id=session_id,
        )
        return run_request

    async def handle_run_flow(self, session_id: str, run_request: RunRequest, flow_runner: FlowRunner) -> None:
        """Handle a flow run request
        
        Args:
            session_id: The session ID
            data: The request data
            flow_runner: The flow runner instance

        """
        try:
            # Reset session data
            self.session_data[session_id] = {
                "messages": [],
                "progress": {"current_step": 0, "total_steps": 100, "status": "waiting"},
                "callback": None,
                "callback_to_ui": run_request.callback_to_ui,
                "outcomes_version": None,
            }

            # Run the flow
            self.session_data[session_id]["callback"] = await flow_runner.run_flow(
                flow_name=run_request.flow,
                run_request=run_request,
            )

        except Exception as e:
            logger.error(f"Error starting flow: {e}")
            await self.send_message(
                session_id,
                ErrorEvent(source="websocket_manager", content=f"Error starting flow: {e!s}"),
            )

    async def handle_user_input(self, session_id: str, data: dict[str, Any]) -> None:
        """Handle user input
        
        Args:
            session_id: The session ID
            data: The user input data

        """
        try:
            # Validate the request
            request = ManagerMessage(**data)

            session = self.session_data.get(session_id)

            # Use the request directly as the message
            await session["callback"](request)

        except Exception as e:
            logger.error(f"Error handling user input: {e}")
            await self.send_message(
                session_id,
                ErrorEvent(source="websocket_manager", content=f"Error handling user input: {e!s}"),
            )
