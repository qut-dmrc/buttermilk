import asyncio
import datetime
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from buttermilk._core.log import logger
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

@flow_data_router.get("/api/session")
async def get_session_endpoint(
    request: Request,
    flows: Annotated[FlowRunner, Depends(get_flows)],
    session_id: str = Query(None, description="Optional existing session ID to validate"),
):
    """Get or create a session for WebSocket connection."""
    logger.debug(f"Session request received. Existing ID: {session_id}")
    
    try:
        # If a session_id is provided, check if it exists
        if session_id and hasattr(flows, 'session_manager'):
            # Check if session exists and is valid
            if session_id in flows.session_manager.sessions:
                session = flows.session_manager.sessions[session_id]
                if session.status.value in ["active", "initializing"]:
                    logger.info(f"Returning existing session: {session_id}")
                    return JSONResponse({
                        "sessionId": session_id,
                        "status": session.status.value,
                        "created_at": session.created_at.isoformat() if hasattr(session, 'created_at') else None
                    })
        
        # Create a new session ID
        new_session_id = str(uuid.uuid4())
        logger.info(f"Creating new session: {new_session_id}")
        
        # Pre-create the session in the session manager if it exists
        # This ensures the WebSocket connection will find it
        if hasattr(flows, 'session_manager'):
            # The session will be fully initialized when WebSocket connects
            # For now, just return the ID
            pass
        
        return JSONResponse({
            "sessionId": new_session_id,
            "status": "new",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in session endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create session")

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


@flow_data_router.get("/api/flows/{flow}/records")
async def get_records_list_endpoint_flow_only(
    request: Request,
    flows: Annotated[FlowRunner, Depends(get_flows)],
    templates: Annotated[Jinja2Templates, Depends(get_templates)],
    flow: str = Path(..., description="The flow name"),
    include_scores: bool = Query(False, description="Include summary scores in list view"),
):
    """Get records for a specific flow"""
    return await _get_records_impl(request, flows, templates, flow, None, include_scores)


@flow_data_router.get("/api/flows/{flow}/datasets/{dataset}/records")
async def get_records_list_endpoint_with_dataset(
    request: Request,
    flows: Annotated[FlowRunner, Depends(get_flows)],
    templates: Annotated[Jinja2Templates, Depends(get_templates)],
    flow: str = Path(..., description="The flow name"),
    dataset: str = Path(..., description="The dataset name"),
    include_scores: bool = Query(False, description="Include summary scores in list view"),
):
    """Get records for a specific flow and dataset"""
    return await _get_records_impl(request, flows, templates, flow, dataset, include_scores)


async def _get_records_impl(
    request: Request,
    flows: FlowRunner,
    templates: Jinja2Templates,
    flow: str,
    dataset: str | None,
    include_scores: bool,
):
    """Enhanced records list with optional score summaries"""
    accept_header = request.headers.get("accept", "")
    logger.info(f"Records list request received for flow: {flow}, dataset: {dataset}, include_scores: {include_scores} (Accept: {accept_header})")

    if not flow:
        logger.warning("Request to /api/flows/{flow}/records missing 'flow' path parameter.")
        error_msg = "Missing 'flow' path parameter."
        if "application/json" in accept_header:
            return JSONResponse(content={"error": error_msg}, status_code=400)
        context_data = {"records": [], "error": error_msg}
        return await negotiate_response(
            request, context_data, "partials/records_list.html", templates, status_code=400,
        )

    # If no dataset specified, return available datasets instead of records
    if not dataset:
        try:
            available_datasets = await DataService.get_datasets_for_flow(flow, flows)
            logger.debug(f"Returning {len(available_datasets)} available datasets for flow {flow}")
            
            if "application/json" in accept_header:
                return JSONResponse(content={
                    "error": "dataset parameter required",
                    "available_datasets": available_datasets,
                    "message": f"Please specify a dataset. Available options: {', '.join(available_datasets)}"
                }, status_code=400)
            
            context_data = {
                "records": [], 
                "error": f"Dataset parameter required. Available datasets: {', '.join(available_datasets)}",
                "available_datasets": available_datasets,
                "flow": flow
            }
            return await negotiate_response(request, context_data, "partials/records_list.html", templates, status_code=400)
            
        except Exception as e:
            logger.error(f"Error getting datasets for flow {flow}: {e}", exc_info=True)
            error_content = {"error": f"Error getting datasets for flow {flow}: {e!s}"}
            return JSONResponse(content=error_content, status_code=500)

    try:
        records = await DataService.get_records_for_flow(flow, flows, include_scores=include_scores, dataset_name=dataset)
        logger.debug(f"Returning data for {len(records)} records")

        if "application/json" in accept_header:
            # Send native Record objects using Pydantic's model_dump()
            records_data = [record.model_dump() for record in records]
            return JSONResponse(content=records_data)

        # For HTML response, use Pydantic model_dump() as well
        records_data = [record.model_dump() for record in records]
        context_data = {"records": records_data, "flow": flow, "dataset": dataset, "include_scores": include_scores}
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


@flow_data_router.get("/api/flows/{flow}/info")
async def get_flowinfo_endpoint(
    request: Request,
    flows: Annotated[FlowRunner, Depends(get_flows)],
    templates: Annotated[Jinja2Templates, Depends(get_templates)],
    flow: str = Path(..., description="The flow name"),
):
    accept_header = request.headers.get("accept", "")
    logger.info(f"Flow data request received for flow: {flow} (Accept: {accept_header})")

    if not flow:
        logger.warning("Request to /api/flowinfo/ missing 'flow' parameter.")
        error_msg = "Missing 'flow' parameter."
        context_data = {"criteria": [], "record_ids": [], "error": error_msg}
        return await negotiate_response(
            request, context_data, "partials/flow_dependent_data.html", templates, status_code=400,
        )

    try:
        criteria = await DataService.get_criteria_for_flow(flow, flows)
        records = await DataService.get_records_for_flow(flow, flows, include_scores=False)
        models = await DataService.get_models_for_flow(flow, flows)
        datasets = await DataService.get_datasets_for_flow(flow, flows)
        logger.debug(f"Returning data for {len(criteria)} criteria options, {len(records)} record options, and {len(datasets)} dataset options")
        # Use model_dump() to serialize Record objects
        record_data = [record.model_dump() for record in records]
        context_data = {"criteria": criteria, "record_ids": record_data, "models": models, "datasets": datasets}
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


@flow_data_router.get("/api/flows/{flow}/records/{record_id}")
async def get_record_endpoint_flow_only(
    record_id: str = Path(..., description="The record ID"),
    flow: str = Path(..., description="The flow name for data context"),
    flows: Annotated[FlowRunner, Depends(get_flows)] = None,
):
    """Get individual record details for flow only"""
    return await _get_record_impl(record_id, flow, None, flows)


@flow_data_router.get("/api/flows/{flow}/datasets/{dataset}/records/{record_id}")
async def get_record_endpoint_with_dataset(
    record_id: str = Path(..., description="The record ID"),
    flow: str = Path(..., description="The flow name for data context"),
    dataset: str = Path(..., description="The dataset name"),
    flows: Annotated[FlowRunner, Depends(get_flows)] = None,
):
    """Get individual record details with dataset"""
    return await _get_record_impl(record_id, flow, dataset, flows)


async def _get_record_impl(
    record_id: str,
    flow: str,
    dataset: str | None,
    flows: FlowRunner,
):
    """Get individual record details"""
    logger.debug(f"Request received for /api/flows/{flow}/records/{record_id} with dataset {dataset}")

    if not flow or flow.strip() == "":
        raise HTTPException(status_code=422, detail="Missing 'flow' path parameter")

    if not record_id or record_id.strip() == "" or record_id == "undefined":
        raise HTTPException(status_code=422, detail="Invalid record_id parameter")

    if flow not in flows.flows:
        raise HTTPException(status_code=422, detail=f"Invalid flow: {flow}")

    try:
        record = await DataService.get_record_by_id(record_id, flow, flows, dataset_name=dataset)

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


@flow_data_router.get("/api/flows/{flow}/records/{record_id}/scores")
async def get_record_scores_endpoint_flow_only(
    record_id: str = Path(..., description="The record ID"),
    flow: str = Path(..., description="The flow name for data context"),
    session_id: str = Query(None, description="Optional session ID for filtering"),
    flows: Annotated[FlowRunner, Depends(get_flows)] = None,
):
    """Get toxicity scores for a specific record (flow only)"""
    return await _get_record_scores_impl(record_id, flow, None, session_id, flows)


@flow_data_router.get("/api/flows/{flow}/datasets/{dataset}/records/{record_id}/scores")
async def get_record_scores_endpoint_with_dataset(
    record_id: str = Path(..., description="The record ID"),
    flow: str = Path(..., description="The flow name for data context"),
    dataset: str = Path(..., description="The dataset name"),
    session_id: str = Query(None, description="Optional session ID for filtering"),
    flows: Annotated[FlowRunner, Depends(get_flows)] = None,
):
    """Get toxicity scores for a specific record with dataset"""
    return await _get_record_scores_impl(record_id, flow, dataset, session_id, flows)


async def _get_record_scores_impl(
    record_id: str,
    flow: str,
    dataset: str | None,
    session_id: str | None,
    flows: FlowRunner,
):
    """Get toxicity scores for a specific record"""
    logger.debug(f"Request received for /api/flows/{flow}/records/{record_id}/scores with session: {session_id}")

    if not flow or flow.strip() == "":
        raise HTTPException(status_code=422, detail="Missing 'flow' path parameter")

    if not record_id or record_id.strip() == "" or record_id == "undefined":
        raise HTTPException(status_code=422, detail="Invalid record_id parameter")

    if flow not in flows.flows:
        raise HTTPException(status_code=422, detail=f"Invalid flow: {flow}")

    try:
        agent_traces = await DataService.get_scores_for_record(record_id, flow, flows, session_id)

        # Send native AgentTrace objects directly using Pydantic's model_dump()
        scores_data = {
            "record_id": record_id,
            "agent_traces": [trace.model_dump() for trace in agent_traces]
        }

        return JSONResponse(content=scores_data)

    except Exception as e:
        logger.error(f"Error getting scores for record {record_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving scores")


@flow_data_router.get("/api/flows/{flow}/records/{record_id}/responses")
async def get_record_responses_endpoint_flow_only(
    record_id: str = Path(..., description="The record ID"),
    flow: str = Path(..., description="The flow name for data context"),
    session_id: str = Query(None, description="Optional session ID for filtering"),
    include_reasoning: bool = Query(True, description="Include detailed reasoning"),
    flows: Annotated[FlowRunner, Depends(get_flows)] = None,
):
    """Get detailed AI responses for a specific record (flow only)"""
    return await _get_record_responses_impl(record_id, flow, None, session_id, include_reasoning, flows)


@flow_data_router.get("/api/flows/{flow}/datasets/{dataset}/records/{record_id}/responses")
async def get_record_responses_endpoint_with_dataset(
    record_id: str = Path(..., description="The record ID"),
    flow: str = Path(..., description="The flow name for data context"),
    dataset: str = Path(..., description="The dataset name"),
    session_id: str = Query(None, description="Optional session ID for filtering"),
    include_reasoning: bool = Query(True, description="Include detailed reasoning"),
    flows: Annotated[FlowRunner, Depends(get_flows)] = None,
):
    """Get detailed AI responses for a specific record with dataset"""
    return await _get_record_responses_impl(record_id, flow, dataset, session_id, include_reasoning, flows)


async def _get_record_responses_impl(
    record_id: str,
    flow: str,
    dataset: str | None,
    session_id: str | None,
    include_reasoning: bool,
    flows: FlowRunner,
):
    """Get detailed AI responses for a specific record"""
    logger.debug(f"Request received for /api/flows/{flow}/records/{record_id}/responses with session: {session_id}")

    if not flow or flow.strip() == "":
        raise HTTPException(status_code=422, detail="Missing 'flow' path parameter")

    if not record_id or record_id.strip() == "" or record_id == "undefined":
        raise HTTPException(status_code=422, detail="Invalid record_id parameter")

    if flow not in flows.flows:
        raise HTTPException(status_code=422, detail=f"Invalid flow: {flow}")

    try:
        agent_traces = await DataService.get_responses_for_record(record_id, flow, flows, session_id, include_reasoning)

        # Send native AgentTrace objects directly using Pydantic's model_dump()
        responses_data = {
            "record_id": record_id,
            "agent_traces": [trace.model_dump() for trace in agent_traces]
        }

        return JSONResponse(content=responses_data)

    except Exception as e:
        logger.error(f"Error getting responses for record {record_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving responses")


# --- Score Page Routes ---

@flow_data_router.get("/score/{flow}/{record_id}")
async def get_score_page_endpoint_flow_only(
    request: Request,
    record_id: str = Path(..., description="The record ID"),
    flow: str = Path(..., description="The flow name"),
    templates: Annotated[Jinja2Templates, Depends(get_templates)] = None,
    flows: Annotated[FlowRunner, Depends(get_flows)] = None,
):
    """Score page route using flow only"""
    return await _get_score_page_impl(request, record_id, flow, None, templates, flows)


@flow_data_router.get("/score/{flow}/{dataset}/{record_id}")
async def get_score_page_endpoint_with_dataset(
    request: Request,
    record_id: str = Path(..., description="The record ID"),
    flow: str = Path(..., description="The flow name"),
    dataset: str = Path(..., description="The dataset name"),
    templates: Annotated[Jinja2Templates, Depends(get_templates)] = None,
    flows: Annotated[FlowRunner, Depends(get_flows)] = None,
):
    """Score page route using flow and dataset"""
    return await _get_score_page_impl(request, record_id, flow, dataset, templates, flows)


async def _get_score_page_impl(
    request: Request,
    record_id: str,
    flow: str,
    dataset: str | None,
    templates: Jinja2Templates,
    flows: FlowRunner,
):
    """Score page route that uses path templates instead of query parameters"""
    logger.debug(f"Request received for /score/{flow}/{dataset or ''}/{record_id}")

    if not flow or flow.strip() == "":
        raise HTTPException(status_code=422, detail="Missing 'flow' path parameter")

    if not record_id or record_id.strip() == "" or record_id == "undefined":
        raise HTTPException(status_code=422, detail="Invalid record_id parameter")

    if flow not in flows.flows:
        raise HTTPException(status_code=422, detail=f"Invalid flow: {flow}")

    try:
        # This is an HTML route that will load the score page
        # The frontend JavaScript will then fetch data via the API endpoints
        context_data = {
            "flow": flow,
            "dataset": dataset,
            "record_id": record_id
        }

        return templates.TemplateResponse(
            "score.html",
            {"request": request, **context_data}
        )

    except Exception as e:
        logger.error(f"Error loading score page for record {record_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error loading score page")
