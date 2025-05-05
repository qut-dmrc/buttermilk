import datetime
from functools import wraps

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from buttermilk.api.services.data_service import DataService
from buttermilk.api.services.websocket_service import WebSocketManager
from buttermilk.bm import logger


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

    def _negotiate_content(self, html_template_name: str):
        """Decorator to handle content negotiation (HTML vs JSON).

        Args:
            html_template_name: The name of the Jinja2 template for HTML responses.

        """
        def decorator(func):
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                # Call the original route function, passing self
                # It should return a dictionary for context/JSON,
                # or a full Response object (e.g., for errors).
                result = await func(self, request, *args, **kwargs)

                # If the route returned a full Response, just return it
                if isinstance(result, (JSONResponse, HTMLResponse, Response)):
                    return result

                # Otherwise, assume it's a dict for context/JSON
                context_data = result
                accept_header = request.headers.get("accept", "")

                if "text/html" in accept_header:
                    logger.debug(f"Returning HTML template '{html_template_name}' based on Accept header for {request.url.path}")
                    return self.templates.TemplateResponse(
                        html_template_name,
                        {"request": request, **context_data},
                    )
                # Default to JSON
                logger.debug(f"Returning JSON based on Accept header for {request.url.path}")
                return JSONResponse(content=context_data)
            return wrapper
        return decorator

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
        async def dashboard(self, request: Request):
            """Render the dashboard page with charts and data tables"""
            return self.templates.TemplateResponse(
                "dashboard.html",
                {"request": request},
            )

        @self.router.get("/api/debug/", response_class=HTMLResponse)  # Keep this as HTML only
        async def get_debug(self, request: Request):
            """Debug endpoint to test template rendering"""
            # import datetime # Moved to top
            return self.templates.TemplateResponse(
                "partials/debug.html",
                {"request": request, "now": datetime.datetime.now()},
            )

        @self.router.get("/api/outcomes/")
        @self._negotiate_content("partials/outcomes_panel.html")
        async def get_outcomes(self, request: Request):  # Add self
            """Get outcomes data (predictions and scores).
            Returns JSON by default or HTML if 'text/html' is accepted.
            """
            session_id = request.query_params.get("session_id")
            client_version = request.query_params.get("version", "0")  # For 304 check
            accept_header = request.headers.get("accept", "")
            logger.debug(f"Request received for /api/outcomes/ (Session: {session_id}, Accept: {accept_header})")

            session_data = DataService.safely_get_session_data(self.websocket_manager, session_id or "")
            pending_agents = session_data.get("pending_agents", [])
            current_version = "0"

            if session_id and session_id in self.websocket_manager.session_data:
                session_state = self.websocket_manager.session_data[session_id]
                current_version = getattr(session_state, "outcomes_version", "0") or "0"

                # Handle 304 Not Modified - return Response directly
                if client_version == current_version:
                    logger.debug(f"Client version {client_version} matches current {current_version}. Returning 304.")
                    return Response(status_code=304)

                if hasattr(session_state, "messages"):
                    original_messages = [msg_data["original"] for msg_data in session_state.messages if "original" in msg_data]

            # Return data dict for decorator
            return {"pending_agents": pending_agents}

        @self.router.get("/api/history/")
        @self._negotiate_content("partials/run_history.html")  # Apply decorator
        async def get_run_history(self, request: Request):  # Add self
            """Get run history for a specific flow, criteria, and record.
            Returns JSON by default or HTML if 'text/html' is accepted.
            """
            flow_name = request.query_params.get("flow")
            criteria = request.query_params.get("criteria")
            record_id = request.query_params.get("record_id")
            accept_header = request.headers.get("accept", "")
            logger.debug(f"Request received for /api/history/ (Flow: {flow_name}, Criteria: {criteria}, Record: {record_id} Accept: {accept_header})")

            if not flow_name or not criteria or not record_id:
                logger.warning("Request to /api/history/ missing required query parameters.")
                # Return specific error responses directly
                error_msg = "Missing required parameters: flow, criteria, and record_id"
                if "text/html" in accept_header:
                    return self.templates.TemplateResponse(
                        "partials/error.html",  # Assuming you have this
                        {"request": request, "error": error_msg},
                        status_code=400,
                    )
                return JSONResponse(content={"error": error_msg}, status_code=400)

            history = await DataService.get_run_history(flow_name, criteria, record_id, self.flows)

            # Handle empty history - return data dict, let template handle display
            # Or return specific template if preferred for empty HTML?
            # Let's return the data and let the main template handle empty case
            # if not history and "text/html" in accept_header:
            #     logger.debug("No history found, returning empty history partial.")
            #     # If you want a specific template for empty HTML response:
            #     return self.templates.TemplateResponse(
            #         "partials/empty_history.html", # Assuming you have this
            #         {"request": request},
            #     )

            # Return data dict for decorator
            return {"history": history}

    def get_router(self) -> APIRouter:
        """Get the router instance
        
        Returns:
            APIRouter: The FastAPI router

        """
        return self.router
