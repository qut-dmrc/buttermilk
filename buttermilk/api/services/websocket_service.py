import time
from collections.abc import Callable
from typing import Any

from fastapi import WebSocket

from buttermilk._core.contract import (
    ErrorEvent,
    ManagerRequest,
    ManagerResponse,
    TaskProcessingComplete,
    TaskProgressUpdate,
)
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.types import RunRequest
from buttermilk.bm import logger

# Session data structure keys
# - messages: list of message objects
# - progress: dict with progress information
# - callback: callable for sending messages back to the flow
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
        self.session_data[session_id] = dict(
            messages=[],
            progress={"current_step": 0, "total_steps": 100, "status": "waiting"},
            callback=None,
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

    async def send_message(self, session_id: str, message: Any | dict[str, Any]) -> None:
        """Send a message to a WebSocket connection
        
        Args:
            session_id: The session ID
            message: The message to send

        """
        if session_id not in self.active_connections:
            logger.warning(f"Attempted to send message to non-existent session: {session_id}")
            return

        try:
            # Convert to dict if it's a Pydantic model
            if not isinstance(message, dict):
                if hasattr(message, "model_dump"):  # Pydantic v2
                    message_data = message.model_dump(mode="json")
                else:
                    # Convert to dict for JSON serialization
                    message_data = {"content": str(message), "type": "unknown"}
            else:
                message_data = message

            await self.active_connections[session_id].send_json(message_data)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            try:
                error_data = ErrorEvent(source="websocket_manager", content=f"Failed to send message: {e!s}")
                await self.active_connections[session_id].send_json(error_data)
            except:
                logger.error("Failed to send error message")

    async def process_message(self, session_id: str, message: dict[str, Any], flow_runner) -> None:
        """Process a message from a WebSocket connection
        
        Args:
            session_id: The session ID
            message: The message to process
            flow_runner: The flow runner instance

        """
        try:
            message_type = message.get("type")

            if message_type == "run_flow":
                await self.handle_run_flow(session_id, message, flow_runner)
            elif message_type == "user_input":
                await self.handle_user_input(session_id, message)
            elif message_type == "confirm":
                await self.handle_confirm(session_id)
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

    async def handle_run_flow(self, session_id: str, data: dict[str, Any], flow_runner) -> None:
        """Handle a flow run request
        
        Args:
            session_id: The session ID
            data: The request data
            flow_runner: The flow runner instance

        """
        try:
            # Validate the request
            request = RunRequest(session_id=session_id, **data)

            if not request.flow or not request.record_id:
                await self.send_message(
                    session_id,
                    ErrorEvent(source="websocket_manager", content="Missing required fields: flow and record_id"),
                )
                return

            # Create run request
            parameters = {}

            # Extract criteria from the original data if present
            if "criteria" in data:
                parameters["criteria"] = data["criteria"]
            else:
                raise ProcessingError("You must add criteria params to the run request.")

            run_request = RunRequest(
                flow=request.flow,
                record_id=request.record_id,
                parameters=parameters,
                callback_to_ui=self.callback_to_ui(session_id),
                session_id=session_id,
            )

            # Reset session data
            self.session_data[session_id] = {
                "messages": [],
                "progress": {"current_step": 0, "total_steps": 100, "status": "waiting"},
                "callback": None,
                "outcomes_version": None,
            }

            # Run the flow
            self.session_data[session_id]["callback"] = await flow_runner.run_flow(
                flow_name=request.flow,
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
            request = ManagerResponse(**data)

            session = self.session_data.get(session_id)
            if not session or not session.get("callback"):
                await self.send_message(
                    session_id,
                    ErrorEvent(source="websocket_manager", content="No active flow to send message to"),
                )
                return

            # Use the request directly as the message
            await session["callback"](request)

            # Send message to client
            await self.send_message(
                session_id,
                {
                    "type": "message_sent",
                    "message": request,
                },
            )
        except Exception as e:
            logger.error(f"Error handling user input: {e}")
            await self.send_message(
                session_id,
                ErrorEvent(source="websocket_manager", content=f"Error handling user input: {e!s}"),
            )

    async def handle_confirm(self, session_id: str) -> None:
        """Handle confirm action
        
        Args:
            session_id: The session ID

        """
        try:
            session = self.session_data.get(session_id)
            if not session or not session.get("callback"):
                await self.send_message(
                    session_id,
                    ErrorEvent(source="websocket_manager", content="No active flow to confirm"),
                )
                return

            message = ManagerResponse(confirm=True)
            await session["callback"](message)

            # Send confirmation to client
            await self.send_message(
                session_id,
                {
                    "type": "confirmed",
                },
            )
        except Exception as e:
            logger.error(f"Error handling confirm: {e}")
            await self.send_message(
                session_id,
                ErrorEvent(source="websocket_manager", content=f"Error handling confirm: {e!s}"),
            )

    async def send_formatted_message(self, session_id: str, message: Any) -> None:
        """Process and send a message to the client
        
        Args:
            session_id: The session ID
            message: The message from the flow (a Pydantic object)

        """
        if not message:
            return  # Skip empty messages

        try:
            # Store only the original message in session history
            if session_id in self.session_data:
                # Create a message data entry with just the original message
                message_data = {
                    "type": type(message).__name__,
                    "message": message,  # Store the original object
                }
                self.session_data[session_id]["messages"].append(message_data)

                # Update outcomes version for judge messages and score updates
                from buttermilk.agents.evaluators.scorer import QualResults
                from buttermilk.agents.judge import JudgeReasons

                # Check the message's outputs directly
                if (hasattr(message, "outputs") and
                    (isinstance(message.outputs, JudgeReasons) or
                     isinstance(message.outputs, QualResults))):
                    # Generate a new version number (timestamp-based for uniqueness)
                    self.session_data[session_id]["outcomes_version"] = str(int(time.time() * 1000))

            # Send the message directly to the client
            await self.send_message(session_id, message)

        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            # Attempt to send error notification
            try:
                await self.send_message(
                    session_id,
                    ErrorEvent(source="websocket_manager", content=f"Failed to process message: {e!s}"),
                )
            except:
                logger.error("Failed to send error message to client")

    def callback_to_ui(self, session_id: str) -> Callable:
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
            # Send message to client if we have an active connection
            if session_id in self.active_connections:
                # For TaskProgressUpdate, we need to update the progress tracking in session data
                if isinstance(message, TaskProgressUpdate) and session_id in self.session_data:
                    # Create progress data dictionary from the TaskProgressUpdate object directly
                    progress_data = {
                        "role": message.role,
                        "step_name": message.step_name,
                        "status": message.status,
                        "message": message.message,
                        "total_steps": message.total_steps,
                        "current_step": message.current_step,
                        "pending_agents": [],
                    }

                    # Update session data - add pending agents to the progress data
                    if session_id in self.session_data:
                        # Get existing progress data
                        existing_progress = self.session_data[session_id]["progress"]

                        # Initialize pending agents list if it doesn't exist
                        if "pending_agents" not in existing_progress:
                            existing_progress["pending_agents"] = []

                        # Copy pending agents to new progress data
                        progress_data["pending_agents"] = existing_progress.get("pending_agents", [])

                        # Update pending agents based on status
                        if message.status == "started" and message.role not in progress_data["pending_agents"]:
                            progress_data["pending_agents"].append(message.role)
                        elif message.status == "completed" and message.role in progress_data["pending_agents"]:
                            progress_data["pending_agents"].remove(message.role)

                        # Update session data
                        self.session_data[session_id]["progress"] = progress_data

                    # Create and send a progress update event
                    update_event = {
                        "type": "progress_update",
                        "progress": progress_data,
                    }
                    await self.send_message(session_id, update_event)
                else:
                    # For all other message types, just send them directly
                    await self.send_formatted_message(session_id, message)

                # For special message types that indicate state changes, send an additional state notification
                if isinstance(message, (TaskProcessingComplete, ManagerRequest, ErrorEvent)):
                    await self.send_message(
                        session_id,
                        dict(state=type(message).__name__),
                    )

        return callback
