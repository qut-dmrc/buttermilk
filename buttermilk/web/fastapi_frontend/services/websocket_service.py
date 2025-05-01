import time
from collections.abc import Callable
from typing import Any

from fastapi import WebSocket

from buttermilk._core.agent import ManagerRequest
from buttermilk._core.contract import ErrorEvent, ManagerResponse, TaskProcessingComplete, TaskProgressUpdate
from buttermilk.bm import logger
from buttermilk.web.fastapi_frontend.schemas import (
    ErrorEvent as SchemaErrorEvent,
    FlowStartedEvent,
    FlowStateChangeEvent,
    RequiresConfirmationEvent,
    RunFlowRequest,
    SessionData,
    UserInputRequest,
    WebSocketMessage,
)
from buttermilk.web.fastapi_frontend.services.message_service import MessageService


class WebSocketManager:
    """Manages WebSocket connections and message handling"""

    def __init__(self):
        """Initialize the WebSocket manager"""
        self.active_connections: dict[str, WebSocket] = {}
        self.session_data: dict[str, SessionData] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept a WebSocket connection and store it
        
        Args:
            websocket: The WebSocket connection
            session_id: The session ID

        """
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_data[session_id] = SessionData(
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

    async def send_message(self, session_id: str, message: WebSocketMessage | dict[str, Any]) -> None:
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
            data = None
            if not isinstance(message, dict):
                if hasattr(message, "model_dump"):  # Pydantic v2
                    data = message.model_dump()
                elif hasattr(message, "dict"):  # Pydantic v1 compatibility
                    data = message.dict()
                else:
                    data = message
            else:
                data = message

            await self.active_connections[session_id].send_json(data)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            try:
                error_message = SchemaErrorEvent(message=f"Failed to send message: {e!s}")
                error_data = error_message.dict() if hasattr(error_message, "dict") else {"type": "error", "message": str(e)}
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
                    SchemaErrorEvent(message=f"Unknown message type: {message_type}"),
                )
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.send_message(
                session_id,
                SchemaErrorEvent(message=f"Error processing message: {e!s}"),
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
            request = RunFlowRequest(**data)

            if not request.flow or not request.record_id:
                await self.send_message(
                    session_id,
                    SchemaErrorEvent(message="Missing required fields: flow and record_id"),
                )
                return

            # Create run request
            from buttermilk._core.types import RunRequest
            run_request = RunRequest(
                flow=request.flow,
                record_id=request.record_id,
                parameters=dict(criteria=request.criteria),
                client_callback=self.callback_to_ui(session_id),
                session_id=session_id,
            )

            # Reset session data
            self.session_data[session_id] = SessionData(
                messages=[],
                progress={"current_step": 0, "total_steps": 100, "status": "waiting"},
                callback=None,
                outcomes_version=None,
            )

            # Run the flow
            self.session_data[session_id].callback = await flow_runner.run_flow(
                flow_name=request.flow,
                run_request=run_request,
            )

            # Send confirmation
            await self.send_message(
                session_id,
                FlowStartedEvent(
                    flow=request.flow,
                    record_id=request.record_id,
                ),
            )
        except Exception as e:
            logger.error(f"Error starting flow: {e}")
            await self.send_message(
                session_id,
                SchemaErrorEvent(message=f"Error starting flow: {e!s}"),
            )

    async def handle_user_input(self, session_id: str, data: dict[str, Any]) -> None:
        """Handle user input
        
        Args:
            session_id: The session ID
            data: The user input data

        """
        try:
            # Validate the request
            request = UserInputRequest(**data)

            session = self.session_data.get(session_id)
            if not session or not session.callback:
                await self.send_message(
                    session_id,
                    SchemaErrorEvent(message="No active flow to send message to"),
                )
                return

            await session.callback(request.message)

            # Send message to client
            await self.send_message(
                session_id,
                {
                    "type": "message_sent",
                    "message": request.message,
                },
            )
        except Exception as e:
            logger.error(f"Error handling user input: {e}")
            await self.send_message(
                session_id,
                SchemaErrorEvent(message=f"Error handling user input: {e!s}"),
            )

    async def handle_confirm(self, session_id: str) -> None:
        """Handle confirm action
        
        Args:
            session_id: The session ID

        """
        try:
            session = self.session_data.get(session_id)
            if not session or not session.callback:
                await self.send_message(
                    session_id,
                    SchemaErrorEvent(message="No active flow to confirm"),
                )
                return

            message = ManagerResponse(confirm=True)
            await session.callback(message)

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
                SchemaErrorEvent(message=f"Error handling confirm: {e!s}"),
            )

    async def send_formatted_message(self, session_id: str, message: Any) -> None:
        """Process and send a message to the client with standardized format handling
        
        Args:
            session_id: The session ID
            message: The message to send

        """
        from buttermilk.web.fastapi_frontend.services.message_service import MessageService
        formatted_output = MessageService.format_message_for_client(message)

        if not formatted_output:
            return  # Skip empty messages

        try:
            # Check if the formatted output is already a structured message (dict)
            if isinstance(formatted_output, dict):
                # Store in session data
                if session_id in self.session_data:
                    # Add message to history
                    message_data = {
                        "content": formatted_output,
                        "type": type(message).__name__,
                    }
                    self.session_data[session_id].messages.append(message_data)

                    # Update outcomes version if this is a score or prediction message
                    if formatted_output.get("type") == "score_update" or (
                        formatted_output.get("type") == "chat_message" and
                        formatted_output.get("agent_info", {}).get("role", "").lower() in ["judge", "synthesiser"]
                    ):
                        # Generate a new version number (timestamp-based for uniqueness)
                        self.session_data[session_id].outcomes_version = str(int(time.time() * 1000))

                # Send the structured message directly
                logger.debug(f"Sending structured message of type: {formatted_output.get('type', 'unknown')}")
                await self.send_message(session_id, formatted_output)

            # Handle regular string content (usually HTML)
            elif isinstance(formatted_output, str):
                # Store in session data
                if session_id in self.session_data:
                    # Add message to history
                    message_data = {
                        "content": formatted_output,
                        "type": type(message).__name__,
                    }
                    self.session_data[session_id].messages.append(message_data)

                # Regular message - send as chat message
                await self.send_message(
                    session_id,
                    {
                        "type": "chat_message",
                        "content": formatted_output,
                    },
                )
            else:
                logger.warning(f"Unexpected output type from format_message_for_client: {type(formatted_output)}")

        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            # Attempt to send error notification
            try:
                await self.send_message(
                    session_id,
                    SchemaErrorEvent(message=f"Failed to process message: {e!s}"),
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
                await self.send_formatted_message(session_id, message)

            # Handle progress updates
            if isinstance(message, TaskProgressUpdate) and session_id in self.active_connections:
                # Create progress data dictionary
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
                    existing_progress = self.session_data[session_id].progress

                    # Initialize pending agents list if it doesn't exist
                    if "pending_agents" not in existing_progress:
                        existing_progress["pending_agents"] = []

                    # Copy pending agents to new progress data
                    progress_data["pending_agents"] = existing_progress.get("pending_agents", [])

                    # Save pending_agents if a new agent starts working
                    if message.status == "started" and message.role not in progress_data["pending_agents"]:
                        progress_data["pending_agents"].append(message.role)

                    # Remove from pending list if the agent is the same as the current agent
                    if message.status == "completed" and message.role in progress_data["pending_agents"]:
                        progress_data["pending_agents"].remove(message.role)

                    # Update session data
                    self.session_data[session_id].progress = progress_data

                # Create a ProgressUpdateEvent to send to the client
                try:
                    update_event = {
                        "type": "progress_update",
                        "progress": progress_data,
                    }
                    await self.send_message(session_id, update_event)
                except Exception as e:
                    logger.error(f"Error sending progress update: {e}")

            # Handle completion and interaction states
            if isinstance(message, (TaskProcessingComplete, ManagerRequest, ErrorEvent)) and session_id in self.active_connections:
                await self.send_message(
                    session_id,
                    FlowStateChangeEvent(state=type(message).__name__),
                )

                if isinstance(message, ManagerRequest):
                    await self.send_message(
                        session_id,
                        RequiresConfirmationEvent(),
                    )

        return callback
