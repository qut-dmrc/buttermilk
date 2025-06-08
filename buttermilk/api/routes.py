import asyncio
import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from buttermilk._core.log import logger
from buttermilk._core.types import Record
from buttermilk._core.contract import AgentTrace
from buttermilk.api.services.data_service import DataService

FlowRunner = Any

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


@flow_data_router.get("/api/records")
async def get_records_list_endpoint(
    request: Request,
    flows: Annotated[FlowRunner, Depends(get_flows)],
    templates: Annotated[Jinja2Templates, Depends(get_templates)],
    flow: str = Query(..., description="The flow name"),
    include_scores: bool = Query(False, description="Include summary scores in list view"),
):
    """Enhanced records list with optional score summaries"""
    accept_header = request.headers.get("accept", "")
    logger.info(f"Records list request received for flow: {flow}, include_scores: {include_scores} (Accept: {accept_header})")

    if not flow:
        logger.warning("Request to /api/records missing 'flow' query parameter.")
        error_msg = "Missing 'flow' query parameter."
        if "application/json" in accept_header:
            return JSONResponse(content={"error": error_msg}, status_code=400)
        context_data = {"records": [], "error": error_msg}
        return await negotiate_response(
            request, context_data, "partials/records_list.html", templates, status_code=400,
        )

    try:
        records = await DataService.get_records_for_flow(flow, flows, include_scores=include_scores)
        logger.debug(f"Returning data for {len(records)} records")

        if "application/json" in accept_header:
            # Send native Record objects using Pydantic's model_dump()
            records_data = [record.model_dump() for record in records]
            return JSONResponse(content=records_data)

        # For HTML response, use Pydantic model_dump() as well
        records_data = [record.model_dump() for record in records]
        context_data = {"records": records_data, "flow": flow, "include_scores": include_scores}
        return await negotiate_response(request, context_data, "partials/records_list.html", templates)

    except Exception as e:
        logger.error(f"Error getting records for flow {flow}: {e}", exc_info=True)
        error_content = {"error": f"Error getting records for flow {flow}: {e!s}"}
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
        records = await DataService.get_records_for_flow(flow, flows, include_scores=False)
        models = await DataService.get_models_for_flow(flow, flows)
        logger.debug(f"Returning data for {len(criteria)} criteria options and {len(records)} record options")
        # Use model_dump() to serialize Record objects
        record_data = [record.model_dump() for record in records]
        context_data = {"criteria": criteria, "record_ids": record_data, "models": models}
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


# --- New Score Pages API Endpoints ---

@flow_data_router.get("/api/records/{record_id}")
async def get_record_endpoint(
    record_id: str,
    flow: str = Query(..., description="The flow name for data context"),
    flows: Annotated[FlowRunner, Depends(get_flows)] = None,
):
    """Get individual record details"""
    logger.debug(f"Request received for /api/records/{record_id} with flow: {flow}")

    if not flow:
        raise HTTPException(status_code=422, detail="Missing 'flow' query parameter")

    if not record_id or record_id.strip() == "" or record_id == "undefined":
        raise HTTPException(status_code=422, detail="Invalid record_id parameter")

    if flow not in flows.flows:
        raise HTTPException(status_code=422, detail=f"Invalid flow: {flow}")

    try:
        record = await DataService.get_record_by_id(record_id, flow, flows)

        if not record:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Record not found",
                    "detail": f"No record found with id: {record_id} in flow: {flow}",
                    "code": "RECORD_NOT_FOUND"
                }
            )

        # Send native Record object using Pydantic's model_dump()
        return JSONResponse(content=record.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting record {record_id} for flow {flow}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving record")


@flow_data_router.get("/api/records/{record_id}/scores")
async def get_record_scores_endpoint(
    record_id: str,
    flow: str = Query(..., description="The flow name for data context"),
    session_id: str = Query(None, description="Optional session ID for filtering"),
    flows: Annotated[FlowRunner, Depends(get_flows)] = None,
):
    """Get toxicity scores for a specific record"""
    logger.debug(f"Request received for /api/records/{record_id}/scores with flow: {flow}, session: {session_id}")

    if not flow:
        raise HTTPException(status_code=422, detail="Missing 'flow' query parameter")

    if not record_id or record_id.strip() == "" or record_id == "undefined":
        raise HTTPException(status_code=422, detail="Invalid record_id parameter")

    if flow not in flows.flows:
        raise HTTPException(status_code=422, detail=f"Invalid flow: {flow}")

    try:
        agent_traces = await DataService.get_scores_for_record(record_id, flow, session_id)
        
        # Send native AgentTrace objects directly using Pydantic's model_dump()
        scores_data = {
            "record_id": record_id,
            "agent_traces": [trace.model_dump() for trace in agent_traces]
        }
        
        return JSONResponse(content=scores_data)

    except Exception as e:
        logger.error(f"Error getting scores for record {record_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving scores")


@flow_data_router.get("/api/records/{record_id}/responses")
async def get_record_responses_endpoint(
    record_id: str,
    flow: str = Query(..., description="The flow name for data context"),
    session_id: str = Query(None, description="Optional session ID for filtering"),
    include_reasoning: bool = Query(True, description="Include detailed reasoning"),
    flows: Annotated[FlowRunner, Depends(get_flows)] = None,
):
    """Get detailed AI responses for a specific record"""
    logger.debug(f"Request received for /api/records/{record_id}/responses with flow: {flow}, session: {session_id}")

    if not flow:
        raise HTTPException(status_code=422, detail="Missing 'flow' query parameter")

    if not record_id or record_id.strip() == "" or record_id == "undefined":
        raise HTTPException(status_code=422, detail="Invalid record_id parameter")

    if flow not in flows.flows:
        raise HTTPException(status_code=422, detail=f"Invalid flow: {flow}")

    try:
        agent_traces = await DataService.get_responses_for_record(record_id, flow, session_id, include_reasoning)
        
        # Send native AgentTrace objects directly using Pydantic's model_dump()
        responses_data = {
            "record_id": record_id,
            "agent_traces": [trace.model_dump() for trace in agent_traces]
        }
        
        return JSONResponse(content=responses_data)

    except Exception as e:
        logger.error(f"Error getting responses for record {record_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving responses")
