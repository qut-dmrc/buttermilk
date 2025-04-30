import json
import random
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from buttermilk._core.agent import (
    ManagerRequest,
)
from buttermilk._core.contract import (
    ErrorEvent,
    ManagerResponse,
    TaskProcessingComplete,
    TaskProgressUpdate,
)
from buttermilk._core.types import RunRequest
from buttermilk.bm import logger
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.web.messages import _format_message_for_client

# Define the base directory for the FastAPI app
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


class DashboardApp:
    """Qualitative Data Science Dashboard with LLM Chatbots"""

    def __init__(self, flows):
        """Initialize the dashboard application
        
        Args:
            flows: FlowRunner or MockFlowRunner instance with workflow configurations

        """
        self.app = FastAPI()
        self.flows = flows
        self.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
        self.active_connections: dict[str, WebSocket] = {}
        self.session_data: dict[str, dict[str, Any]] = {}

        # Set up static files
        self.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        # Register routes
        self.setup_routes()

    def setup_routes(self):
        """Set up the application routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            """Render the main dashboard page"""
            flow_choices = list(self.flows.flows.keys())
            session_id = str(uuid.uuid4())

            return self.templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "flow_choices": flow_choices,
                    "session_id": session_id,
                },
            )

        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Render the dashboard page with charts and data tables"""
            return self.templates.TemplateResponse(
                "dashboard.html",
                {"request": request},
            )

        @self.app.get("/api/flows", response_class=HTMLResponse)
        async def get_flows(request: Request):
            """Get all available flows as HTML select options for htmx replacement"""
            flow_choices = list(self.flows.flows.keys())
            return self.templates.TemplateResponse(
                "partials/flow_options.html",
                {"request": request, "flow_choices": flow_choices},
            )

        @self.app.get("/api/criteria/", response_class=HTMLResponse)
        async def get_criteria(request: Request):
            """Get criteria options for a specific flow using query parameter"""
            flow_name = request.query_params.get("flow")  # Get flow from query parameter
            criteria = []

            if not flow_name:
                logger.warning("Request to /api/criteria/ missing 'flow' query parameter.")
                return self.templates.TemplateResponse(
                    "partials/criteria_options.html",
                    {"request": request, "criteria": []},
                )

            try:
                criteria = self.flows.flows[flow_name].parameters["criteria"]
                random.shuffle(criteria)
            except Exception as e:
                logger.error(f"Error loading criteria for flow '{flow_name}': {e}")

            return self.templates.TemplateResponse(
                "partials/criteria_options.html",
                {"request": request, "criteria": criteria},
            )

        @self.app.get("/api/records/", response_class=HTMLResponse)
        async def get_records(request: Request):
            """Get record IDs for a specific flow"""
            flow_name = request.query_params.get("flow")  # Assuming original select has name="flow"

            record_ids = []
            if not flow_name:
                logger.warning("Request to /api/records/ missing 'flow' query parameter.")
                return self.templates.TemplateResponse(
                    "partials/record_options.html",
                    {"request": request, "record_ids": []},
                )

            try:
                if flow_name in self.flows.flows:
                    flow_obj = self.flows.flows.get(flow_name)
                    # Try to use get_record_ids method if it exists
                    if flow_obj and hasattr(flow_obj, "get_record_ids") and callable(flow_obj.get_record_ids):
                        df = await flow_obj.get_record_ids()

                        # Make sure df has an index attribute before using it
                        if hasattr(df, "index"):
                            record_ids = df.index.tolist()
                        elif isinstance(df, list):
                            record_ids = df
                        else:
                            logger.warning(f"Unexpected return type from get_record_ids: {type(df)}")
                    # Otherwise try to use the data directly
                    elif hasattr(flow_obj, "data") and flow_obj.data:
                        # Check if we have mock data first (simpler approach)
                        if "mock_data" in flow_obj.data and "record_ids" in flow_obj.data["mock_data"]:
                            record_ids = flow_obj.data["mock_data"]["record_ids"]
                        else:
                            try:
                                logger.debug(f"Loading data for flow: {flow_name}")
                                df_dict = await prepare_step_df(flow_obj.data)
                                if df_dict and isinstance(df_dict, dict) and len(df_dict) > 0:
                                    df = list(df_dict.values())[-1]
                                    logger.debug(f"Got DataFrame of type: {type(df)}")
                                    # Check if df has index attribute
                                    if hasattr(df, "index"):
                                        logger.debug(f"DataFrame has index of type: {type(df.index)}")
                                        if hasattr(df.index, "values"):
                                            record_ids = df.index.values.tolist()
                                        else:
                                            record_ids = df.index.tolist()
                                    elif isinstance(df, list):
                                        record_ids = df
                                    elif hasattr(df, "to_dict"):
                                        # For pandas DataFrame-like objects
                                        records_dict = df.to_dict("records")
                                        record_ids = [str(r.get("id", i)) for i, r in enumerate(records_dict)]
                                    else:
                                        logger.warning(f"Unable to extract record_ids from data: {type(df)}")
                            except Exception as e:
                                logger.error(f"Error preparing data: {e}")
            except Exception as e:
                logger.error(f"Error loading data for flow '{flow_name}': {e}")

            return self.templates.TemplateResponse(
                "partials/record_options.html",
                {"request": request, "record_ids": record_ids},
            )

        @self.app.get("/api/history/", response_class=HTMLResponse)
        async def get_run_history(request: Request):
            """Get run history for a specific flow, criteria, and record using query parameters"""
            flow_name = request.query_params.get("flow")
            criteria = request.query_params.get("criteria")
            record_id = request.query_params.get("record_id")

            if not flow_name or not criteria or not record_id:
                logger.warning("Request to /api/history/ missing required query parameters.")
                return self.templates.TemplateResponse(
                    "partials/error.html",
                    {"request": request, "error": "Missing required parameters: flow, criteria, and record_id"},
                )
            """Get run history for a specific flow, criteria, and record"""
            try:
                # Format of the SQL query to get judge and synth runs
                sql = """
                SELECT
                    *
                FROM
                    `prosocial-443205.testing.flow_score_results`
                """
                # Execute the query using bm.run_query
                results_df = self.flows.bm.run_query(sql)

                # If the results are empty
                if not isinstance(results_df, pd.DataFrame) or results_df.empty:
                    return self.templates.TemplateResponse(
                        "partials/empty_history.html",
                        {"request": request},
                    )

                return self.templates.TemplateResponse(
                    "partials/run_history.html",
                    {"request": request, "history": results_df.to_dict("records")},
                )
            except Exception as e:
                logger.error(f"Error loading run history: {e}")
                return self.templates.TemplateResponse(
                    "partials/error.html",
                    {"request": request, "error": str(e)},
                )

        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for real-time communication"""
            await websocket.accept()
            self.active_connections[session_id] = websocket
            self.session_data[session_id] = {
                "messages": [],
                "progress": {"current_step": 0, "total_steps": 100, "status": "waiting"},
            }

            try:
                # Listen for messages from the client
                while True:
                    data = await websocket.receive_json()

                    # Handle different message types
                    if data.get("type") == "run_flow":
                        # Start a flow run
                        await self.handle_run_flow(websocket, session_id, data)
                    elif data.get("type") == "user_input":
                        # Handle user message input
                        await self.handle_user_input(websocket, session_id, data)
                    elif data.get("type") == "confirm":
                        # Handle confirm action
                        await self.handle_confirm(websocket, session_id)

            except WebSocketDisconnect:
                # Remove the connection when client disconnects
                if session_id in self.active_connections:
                    del self.active_connections[session_id]

                if session_id in self.session_data:
                    del self.session_data[session_id]

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

    async def handle_run_flow(self, websocket: WebSocket, session_id: str, data: dict[str, Any]):
        """Handle a flow run request
        
        Args:
            websocket: The WebSocket connection
            session_id: The session ID
            data: The request data

        """
        flow_name = data.get("flow")
        record_id = data.get("record_id")
        criteria = data.get("criteria")

        if not flow_name or not record_id:
            await websocket.send_json({
                "type": "error",
                "message": "Missing required fields: flow and record_id",
            })
            return

        # Create run request
        run_request = RunRequest(
            flow=flow_name,
            record_id=record_id,
            parameters=dict(criteria=criteria),
            client_callback=self.callback_to_ui(session_id),
            session_id=session_id,
        )

        # Start the flow
        try:
            # Reset session data
            self.session_data[session_id] = {
                "messages": [],
                "progress": {"current_step": 0, "total_steps": 100, "status": "started"},
                "callback": None,
            }

            # Run the flow
            self.session_data[session_id]["callback"] = await self.flows.run_flow(
                flow_name=flow_name,
                run_request=run_request,
            )

            # Send confirmation
            await websocket.send_json({
                "type": "flow_started",
                "flow": flow_name,
                "record_id": record_id,
            })
        except Exception as e:
            logger.error(f"Error starting flow: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Error starting flow: {e!s}",
            })

    async def handle_user_input(self, websocket: WebSocket, session_id: str, data: dict[str, Any]):
        """Handle user input
        
        Args:
            websocket: The WebSocket connection
            session_id: The session ID
            data: The user input data

        """
        user_input = data.get("message", "")
        callback = self.session_data.get(session_id, {}).get("callback")

        if callback:
            await callback(user_input)

            # Send message to client
            await websocket.send_json({
                "type": "message_sent",
                "message": user_input,
            })
        else:
            await websocket.send_json({
                "type": "error",
                "message": "No active flow to send message to",
            })

    async def handle_confirm(self, websocket: WebSocket, session_id: str):
        """Handle confirm action
        
        Args:
            websocket: The WebSocket connection
            session_id: The session ID

        """
        callback = self.session_data.get(session_id, {}).get("callback")

        if callback:
            message = ManagerResponse(confirm=True)
            await callback(message)

            # Send confirmation to client
            await websocket.send_json({
                "type": "confirmed",
            })
        else:
            await websocket.send_json({
                "type": "error",
                "message": "No active flow to confirm",
            })

    async def send_message_to_client(self, websocket: WebSocket, message: Any):
        """Process and send a message to the client with improved error handling for scripts."""
        formatted_output = _format_message_for_client(message)
        
        if not formatted_output:
            return  # Skip empty messages
        
        # Check if this is a score script (starts with <script> tag)
        is_score_script = formatted_output.strip().startswith("<script>")
        
        try:
            if is_score_script:
                # Send the script directly as a special message type
                logger.debug("Detected score script, sending directly")
                await websocket.send_json({
                    "type": "script_content",
                    "content": formatted_output
                })
            else:
                # Regular message - send as chat message
                await websocket.send_json({
                    "type": "chat_message", 
                    "html": formatted_output
                })
                
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            # Attempt to send error notification
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Failed to process message: {str(e)}"
                })
            except:
                logger.error("Failed to send error message to client")

    def callback_to_ui(self, session_id: str):
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
            # Format the message for the client
            content = _format_message_for_client(message)

            # If we have a formatted message, send it
            if content and session_id in self.active_connections:
                await self.send_message_to_client(self.active_connections[session_id], message)

                # Store in session data
                if session_id in self.session_data:
                    self.session_data[session_id]["messages"].append({
                        "content": content,
                        "type": type(message).__name__,
                    })

            # Handle progress updates
            if isinstance(message, TaskProgressUpdate) and session_id in self.active_connections:
                progress_data = {
                    "role": message.role,
                    "step_name": message.step_name,
                    "status": message.status,
                    "message": message.message,
                    "total_steps": message.total_steps,
                    "current_step": message.current_step,
                }

                # Update session data
                if session_id in self.session_data:
                    self.session_data[session_id]["progress"] = progress_data

                # Send to client
                await self.active_connections[session_id].send_json({
                    "type": "progress_update",
                    "progress": progress_data,
                })

            # Handle completion and interaction states
            if isinstance(message, (TaskProcessingComplete, ManagerRequest, ErrorEvent)) and session_id in self.active_connections:
                await self.active_connections[session_id].send_json({
                    "type": "flow_state_change",
                    "state": type(message).__name__,
                })

                if isinstance(message, ManagerRequest):
                    await self.active_connections[session_id].send_json({
                        "type": "requires_confirmation",
                    })

        return callback

    def get_app(self):
        """Get the FastAPI application
        
        Returns:
            FastAPI: The FastAPI application

        """
        return self.app


def create_dashboard_app(flows) -> FastAPI:
    """Create and configure the dashboard application
    
    Args:
        flows: FlowRunner or mock runner instance with workflow configurations
        
    Returns:
        FastAPI: The FastAPI application

    """
    dashboard = DashboardApp(flows)
    return dashboard.get_app()
