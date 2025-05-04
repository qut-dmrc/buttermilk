import datetime
from functools import wraps

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from buttermilk.api.services.data_service import DataService
from buttermilk.bm import logger


class FlowRoutes:
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
        """Set up the API data routes"""

        @self.router.get("/api/flows")
        @self._negotiate_content("partials/flow_options.html")
        async def get_flows(self, request: Request):
            """Get all available flows.
            f
            Returns:
                JSON by default or HTML if 'text/html' is accepted.

            """
            logger.debug(f"Request received for /api/flows (Accept: {request.headers.get('accept', '')})")
            flow_choices = list(self.flows.flows.keys())
            return {"flow_choices": flow_choices}

        @self.router.get("/api/flowdata/")
        @self._negotiate_content("partials/flow_dependent_data.html")  # Apply decorator
        async def get_flowinfo(self, request: Request):  # Add self
            """Get criteria options and records for a specific flow.
            Returns JSON by default or HTML if 'text/html' is accepted.
            """
            flow_name = request.query_params.get("flow")
            accept_header = request.headers.get("accept", "")  # Needed for error handling below
            logger.info(f"Flow data request received for flow: {flow_name} (Accept: {accept_header})")

            if not flow_name:
                logger.warning("Request to /api/flowdata/ missing 'flow' query parameter.")
                # Return specific error responses directly (decorator passes them through)
                error_msg = "Missing 'flow' query parameter."
                if "text/html" in accept_header:
                     # Return the template directly for errors for now
                     return self.templates.TemplateResponse(
                         "partials/flow_dependent_data.html",  # Or an error partial
                         {"request": request, "criteria": [], "record_ids": [], "error": error_msg},
                         status_code=400,
                     )
                return JSONResponse(content={"error": error_msg}, status_code=400)

            try:
                criteria = await DataService.get_criteria_for_flow(flow_name, self.flows)
                record_ids = await DataService.get_records_for_flow(flow_name, self.flows)

                logger.debug(f"Returning data for {len(criteria)} criteria options and {len(record_ids)} record options")

                return {
                    "criteria": criteria,
                    "record_ids": record_ids,
                }
            except Exception as e:
                logger.error(f"Error getting data for flow {flow_name}: {e}")
                # Return specific error responses directly
                error_content = {"error": f"Error getting data for flow {flow_name}: {e!s}"}
                if "text/html" in accept_header:
                     return self.templates.TemplateResponse(
                         "partials/debug.html",  # Or a dedicated error partial
                         {
                             "request": request,
                             "now": datetime.datetime.now(),
                             "error": error_content["error"],
                         },
                         status_code=500,
                     )
                return JSONResponse(content=error_content, status_code=500)

    def get_router(self) -> APIRouter:
        """Get the router instance
        
        Returns:
            APIRouter: The FastAPI router

        """
        return self.router
