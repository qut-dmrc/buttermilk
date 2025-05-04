import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

FlowRunner = Any

from buttermilk.api.services.data_service import DataService
from buttermilk.api.services.message_service import MessageService
from buttermilk.bm import logger


# --- Dependency Provider Functions ---
async def get_templates(request: Request) -> Jinja2Templates:
    templates = getattr(request.app.state, "templates", None)
    if templates is None:
        raise RuntimeError("Jinja2Templates not found in app.state.templates")
    return templates


async def get_flows(request: Request) -> FlowRunner:
    flows = getattr(request.app.state, "flow_runner", None)
    if flows is None:
        raise RuntimeError("FlowRunner not found in app.state.flows")
    return flows


async def get_websocket_manager(request: Request):
    manager = getattr(request.app.state, "websocket_manager", None)
    if manager is None:
        raise RuntimeError("WebSocketManager not found in app.state.websocket_manager")
    return manager


# --- Router ---
flow_data_router = APIRouter()


# --- Helper for Content Negotiation ---
async def negotiate_response(
    request: Request,
    context_data: dict,
    template_name: str,
    templates: Jinja2Templates,
    status_code: int = 200,
) -> Response:
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header:
        logger.debug(f"Returning HTML template '{template_name}' based on Accept header for {request.url.path}")
        return templates.TemplateResponse(
            template_name,
            {"request": request, **context_data},
            status_code=status_code,
        )
    logger.debug(f"Returning JSON based on Accept header for {request.url.path}")
    return JSONResponse(content=context_data, status_code=status_code)


# --- Routes ---
@flow_data_router.get("/api/flows")
async def get_flows_endpoint(
    request: Request,
    flows: Annotated[FlowRunner, Depends(get_flows)],
    templates: Annotated[Jinja2Templates, Depends(get_templates)],
):
    logger.debug(f"Request received for /api/flows (Accept: {request.headers.get('accept', '')})")
    try:
        flow_choices = list(flows.flows.keys())
        context_data = {"flow_choices": flow_choices}
        return await negotiate_response(request, context_data, "partials/flow_options.html", templates)
    except Exception as e:
        logger.error(f"Error getting flows: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving flows")


@flow_data_router.get("/api/outcomes")
async def get_outcomes_endpoint(
    request: Request,
    websocket_manager: Annotated[Any, Depends(get_websocket_manager)],
    templates: Annotated[Jinja2Templates, Depends(get_templates)],
):
    """Get outcomes data (predictions and scores).
    Returns JSON by default or HTML if 'text/html' is accepted.
    """
    session_id = request.query_params.get("session_id")
    client_version = request.query_params.get("version", "0")  # For 304 check
    accept_header = request.headers.get("accept", "")
    logger.debug(f"Request received for /api/outcomes/ (Session: {session_id}, Accept: {accept_header})")

    session_data = DataService.safely_get_session_data(websocket_manager, session_id or "")
    scores = {}
    pending_agents = session_data.get("pending_agents", [])
    current_version = "0"

    if session_id and session_id in websocket_manager.session_data:
        # Get the session data for this session
        session = websocket_manager.session_data[session_id]
        current_version = session.get("outcomes_version", "0") or "0"

        # Handle 304 Not Modified - return Response directly
        if client_version == current_version:
            logger.debug(f"Client version {client_version} matches current {current_version}. Returning 304.")
            return Response(status_code=304)

        # Extract scores from messages if available
        if "messages" in session:
            scores = MessageService.extract_scores_from_messages(session["messages"])

    # Prepare the context data for response
    context_data = {"scores": scores, "pending_agents": pending_agents}
    return await negotiate_response(request, context_data, "partials/outcomes_panel.html", templates)


@flow_data_router.get("/api/flowinfo")
async def get_flowinfo_endpoint(
    request: Request,
    flows: Annotated[FlowRunner, Depends(get_flows)],
    templates: Annotated[Jinja2Templates, Depends(get_templates)],
    flow: str | None = None,
):
    accept_header = request.headers.get("accept", "")
    logger.info(f"Flow data request received for flow: {flow} (Accept: {accept_header})")

    if not flow:
        logger.warning("Request to /api/flowinfo/ missing 'flow' query parameter.")
        error_msg = "Missing 'flow' query parameter."
        context_data = {"criteria": [], "record_ids": [], "error": error_msg}
        return await negotiate_response(
            request, context_data, "partials/flow_dependent_data.html", templates, status_code=400,
        )

    try:
        criteria = await DataService.get_criteria_for_flow(flow, flows)
        record_ids = await DataService.get_records_for_flow(flow, flows)
        logger.debug(f"Returning data for {len(criteria)} criteria options and {len(record_ids)} record options")
        context_data = {"criteria": criteria, "record_ids": record_ids}
        return await negotiate_response(request, context_data, "partials/flow_dependent_data.html", templates)

    except Exception as e:
        logger.error(f"Error getting data for flow {flow}: {e}", exc_info=True)
        error_content = {"error": f"Error getting data for flow {flow}: {e!s}"}
        if "text/html" in accept_header:
             return templates.TemplateResponse(
                 "partials/debug.html",
                 {
                     "request": request,
                     "now": datetime.datetime.now(),
                     "error": error_content["error"],
                 },
                 status_code=500,
             )
        return JSONResponse(content=error_content, status_code=500)
