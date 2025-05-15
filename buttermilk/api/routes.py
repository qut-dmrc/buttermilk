import asyncio
import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

FlowRunner = Any

from buttermilk.api.services.data_service import DataService
from buttermilk.bm import BM, logger  # Buttermilk global instance and logger

bm = BM()


#
# curl -v 'http://127.0.0.1:8000/api/pull_task' -H 'accept: application/json'

# --- Dependency Provider Functions ---
async def get_templates(request: Request) -> Jinja2Templates:
    templates = getattr(request.app.state, "templates", None)
    if templates is None:
        raise RuntimeError("Jinja2Templates not found in app.state.templates")
    return templates


async def get_flows(request: Request) -> FlowRunner:
    from buttermilk.runner.flowrunner import FlowRunner as FlowRunner_object
    flows = getattr(request.app.state, "flow_runner", None)
    if flows is None or not isinstance(flows, FlowRunner_object):
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
        models = await DataService.get_models_for_flow(flow, flows)
        logger.debug(f"Returning data for {len(criteria)} criteria options and {len(record_ids)} record options")
        context_data = {"criteria": criteria, "record_ids": record_ids, "models": models}
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


@flow_data_router.get("/api/pull_task")
async def pull_task_endpoint(request: Request) -> StreamingResponse:
    logger.debug(f"Request received for /api/pull_task (Accept: {request.headers.get('accept', '')})")
    try:
        from buttermilk.api.job_queue import JobQueueClient
        run_request = await JobQueueClient().pull_single_task()

        asyncio.create_task(request.app.state.flow_runner.run_flow(
                    run_request=run_request,
                    wait_for_completion=False,
                ))

    except Exception as e:
        logger.error(f"Error pulling task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error pulling task")
