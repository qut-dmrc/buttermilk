
from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from buttermilk.bm import logger
from buttermilk.web.fastapi_frontend.services.data_service import DataService
from buttermilk.web.fastapi_frontend.services.message_service import MessageService
from buttermilk.web.fastapi_frontend.services.websocket_service import WebSocketManager


class DashboardRoutes:
    """API routes for the dashboard application"""

    def __init__(self, templates, flows):
        """Initialize the routes
        
        Args:
            templates: Jinja2Templates instance
            flows: FlowRunner instance

        """
        self.templates = templates
        self.flows = flows
        self.router = APIRouter()
        self.websocket_manager = WebSocketManager()

        self.setup_routes()

    def setup_routes(self):
        """Set up the application routes"""

        @self.router.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            """Render the main dashboard page"""
            flow_choices = list(self.flows.flows.keys())
            import uuid
            session_id = str(uuid.uuid4())

            return self.templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "flow_choices": flow_choices,
                    "session_id": session_id,
                },
            )

        @self.router.get("/dashboard", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Render the dashboard page with charts and data tables"""
            return self.templates.TemplateResponse(
                "dashboard.html",
                {"request": request},
            )

        @self.router.get("/api/flows", response_class=HTMLResponse)
        async def get_flows(request: Request):
            """Get all available flows as HTML select options for htmx replacement"""
            flow_choices = list(self.flows.flows.keys())
            return self.templates.TemplateResponse(
                "partials/flow_options.html",
                {"request": request, "flow_choices": flow_choices},
            )

        @self.router.get("/api/debug/", response_class=HTMLResponse)
        async def get_debug(request: Request):
            """Debug endpoint to test template rendering"""
            import datetime
            return self.templates.TemplateResponse(
                "partials/debug.html",
                {"request": request, "now": datetime.datetime.now()},
            )

        @self.router.get("/api/criteria/", response_class=HTMLResponse)
        async def get_criteria(request: Request):
            """Get criteria options for a specific flow using query parameter"""
            flow_name = request.query_params.get("flow")  # Get flow from query parameter
            
            logger.info(f"Criteria request received for flow: {flow_name}")

            if not flow_name:
                logger.warning("Request to /api/criteria/ missing 'flow' query parameter.")
                return self.templates.TemplateResponse(
                    "partials/criteria_options.html",
                    {"request": request, "criteria": []},
                )

            try:
                criteria = await DataService.get_criteria_for_flow(flow_name, self.flows)
                
                # randomize the order
                import random
                random.shuffle(criteria)
                
                logger.info(f"Returning {len(criteria)} criteria options")
                
                return self.templates.TemplateResponse(
                    "partials/criteria_options.html",
                    {"request": request, "criteria": criteria},
                )
            except Exception as e:
                logger.error(f"Error getting criteria for flow {flow_name}: {e}")
                # Return debug template on error
                import datetime
                return self.templates.TemplateResponse(
                    "partials/debug.html",
                    {
                        "request": request, 
                        "now": datetime.datetime.now(),
                        "error": str(e)
                    },
                )

        @self.router.get("/api/records/", response_class=HTMLResponse)
        async def get_records(request: Request):
            """Get record IDs for a specific flow"""
            flow_name = request.query_params.get("flow")

            if not flow_name:
                logger.warning("Request to /api/records/ missing 'flow' query parameter.")
                return self.templates.TemplateResponse(
                    "partials/record_options.html",
                    {"request": request, "record_ids": []},
                )

            record_ids = await DataService.get_records_for_flow(flow_name, self.flows)

            return self.templates.TemplateResponse(
                "partials/record_options.html",
                {"request": request, "record_ids": record_ids},
            )

        @self.router.get("/api/outcomes/", response_class=HTMLResponse)
        async def get_outcomes(request: Request):
            """Get outcomes data (predictions and scores) for HTMX replacement"""
            session_id = request.query_params.get("session_id")

            # Get client version to support conditional responses
            client_version = request.query_params.get("version", "0")

            # Use DataService to safely get session data with default values
            session_data = DataService.safely_get_session_data(self.websocket_manager, session_id or "")

            # Initialize containers
            scores = {}
            outcomes = []
            pending_agents = session_data.get("pending_agents", [])
            current_version = "0"

            # Check if we have a valid session with an outcomes version
            if session_id and session_id in self.websocket_manager.session_data:
                # Get or create outcomes version
                current_version = getattr(self.websocket_manager.session_data[session_id], "outcomes_version", "0") or "0"

                # If client already has latest version, return 304 Not Modified
                if client_version == current_version:
                    return Response(status_code=304)

                # Extract data from session messages if we have a valid session
                if hasattr(self.websocket_manager.session_data[session_id], "messages"):
                    messages = self.websocket_manager.session_data[session_id].messages

                    # Get scores and predictions using message service
                    scores = MessageService.extract_scores_from_messages(messages)
                    outcomes = MessageService.extract_predictions_from_messages(messages)

            # Return the template response with sanitized data
            return self.templates.TemplateResponse(
                "partials/outcomes_panel.html",
                {"request": request, "scores": scores, "outcomes": outcomes, "pending_agents": pending_agents},
            )

        @self.router.get("/api/history/", response_class=HTMLResponse)
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

            # Get run history
            history = await DataService.get_run_history(flow_name, criteria, record_id, self.flows)

            # If the results are empty
            if not history:
                return self.templates.TemplateResponse(
                    "partials/empty_history.html",
                    {"request": request},
                )

            return self.templates.TemplateResponse(
                "partials/run_history.html",
                {"request": request, "history": history},
            )

        @self.router.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for real-time communication"""
            await self.websocket_manager.connect(websocket, session_id)

            try:
                # Listen for messages from the client
                while True:
                    data = await websocket.receive_json()
                    await self.websocket_manager.process_message(session_id, data, self.flows)

            except WebSocketDisconnect:
                self.websocket_manager.disconnect(session_id)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

    def get_router(self) -> APIRouter:
        """Get the router instance
        
        Returns:
            APIRouter: The FastAPI router

        """
        return self.router
