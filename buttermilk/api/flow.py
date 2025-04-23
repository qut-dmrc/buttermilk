import asyncio
import threading
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Literal

import hydra
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from google.cloud import pubsub
from pydantic import BaseModel

from buttermilk.api.stream import FlowRequest, flow_stream
from buttermilk.bm import BM, bm, logger

from .runs import get_recent_runs

INPUT_SOURCE = "api"
app = FastAPI()
flows = dict()

# curl -X 'POST' 'http://127.0.0.1:8000/flow/simple' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"q": "Democrats are arseholes."}'
# curl -X 'POST' 'http://127.0.0.1:8000/flow/simple' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"text": "Democrats are arseholes."}'
# curl -X 'POST' 'http://127.0.0.1:8000/flow/trans' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"uri": "https://www.city-journal.org/article/what-are-we-doing-to-children"}'
# curl -X 'POST' 'http://127.0.0.1:8000/flow/osb' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"q": "Is it still hate speech if the targeted group is not explicitly named?"}'

# curl -X 'POST' 'http://127.0.0.1:8000/flow/hate' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": ["haiku", "gpt4o"], "criteria": "criteria_ordinary", "video": "gs://dmrc-platforms/test/fyp/tiktok-imane-01.mp4"}'
# curl -X 'POST' 'http://127.0.0.1:8000/flow/hate' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"uri": "https://upload.wikimedia.org/wikipedia/en/b/b9/MagrittePipe.jpg"}'
# curl -X 'POST' 'http://127.0.0.1:8000/flow/judge' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": ["haiku", "gpt4o"], "template":"summarise_osb", "text": "gs://dmrc-platforms/data/osb/FB-UK2RUS24.md"}'
# curl -X 'POST' 'http://127.0.0.1:8000/flow/trans' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"record_id": "betoota_snape_trans"}'


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


@app.api_route("/runs/", methods=["GET", "POST"])
async def get_runs_json(request: Request) -> Sequence:
    runs = get_recent_runs()

    results = [Job(**row) for _, row in runs.iterrows()]

    return results


@app.api_route("/html/runs/", methods=["GET", "POST"])
async def get_runs_html(request: Request) -> HTMLResponse:
    data = get_recent_runs()

    rendered_result = templates.TemplateResponse(
        "runs_html.html",
        {"request": request, "data": data},
    )

    return HTMLResponse(rendered_result.body.decode("utf-8"), status_code=200)


@app.api_route("/flow/{flow_name}", methods=["GET", "POST"])
async def run_flow_json(
    flow_name: str,
    request: Request,
    flow_request: FlowRequest | None = "",
) -> StreamingResponse:
    """Run a flow with provided inputs."""

    # Access state via request.app.state
    if not hasattr(request.app.state, "flows") or flow_name not in request.app.state.flows:
        raise HTTPException(status_code=404, detail="Flow configuration not found or flow name invalid")

    if not hasattr(request.app.state, "bm"):
        raise HTTPException(status_code=500, detail="BM instance not found in app state")

    current_bm = request.app.state.bm
    # Get a copy of the flow config to avoid modifying the state directly
    flow_config = request.app.state.flows[flow_name].copy()
    orchestrator_name = flow_config.get("orchestrator", None)

    orchestrator = None
    if orchestrator_name:
        orchestrator_cls = request.app.state.orchestrators.get(orchestrator_name)
        if orchestrator_cls:
            orchestrator = orchestrator_cls(bm=current_bm, **flow_config)
        else:
            logger.error(f"Unknown orchestrator name specified: {orchestrator_name}")
            raise HTTPException(status_code=500, detail=f"Invalid orchestrator configuration: {orchestrator_name}")
    else:
        raise HTTPException(status_code=500, detail="Orchestrator not specified in flow config")

    return StreamingResponse(
        flow_stream(orchestrator.run(), flow_request),
        media_type="application/json",
    )
    raise HTTPException(status_code=403, detail="Flow not valid")


@app.api_route("/html/flow/{flow}", methods=["GET", "POST"])
@app.api_route("/html/flow", methods=["GET", "POST"])
async def run_route_html(
    request: Request,
    flow: str = "",
    flow_request: FlowRequest | None = "",
) -> StreamingResponse:
    if flow not in flows:
        raise HTTPException(status_code=403, detail="Flow not valid")

    async def result_generator() -> AsyncGenerator[str, None]:
        logger.debug(
            f"Received request for HTML flow {flow} and flow_request {flow_request}",
        )
        try:
            async for data in flow_stream(
                flows[flow],
                flow_request=flow_request,
                return_json=False,
            ):
                # Render the template with the response data
                rendered_result = templates.TemplateResponse(
                    "flow_html.html",
                    {"request": request, "data": data},
                )
                yield rendered_result.body.decode("utf-8")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(result_generator(), media_type="text/html")


# Set up CORS
origins = [
    "http://localhost:5000",  # Frontend running on localhost:5000
    "http://127.0.0.1:5000",
    "http://127.0.0.1:8080",  # Frontend running on localhost:8080
    "http://localhost:8080",
    "http://localhost:8000",  # Allow requests from localhost:8000
    "http://127.0.0.1:8000",
    "http://automod.cc",  # Allow requests from your domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Custom middleware to log CORS failures
@app.middleware("http")
async def log_cors_failures(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin:
        logger.debug(f"CORS check for {origin}")

    response = await call_next(request)
    return response
