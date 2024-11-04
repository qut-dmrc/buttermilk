import asyncio
import itertools
import json
import os
import sys
from pathlib import Path

import threading
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Self, Sequence
from cloudpathlib import AnyPath
from fastapi.concurrency import run_in_threadpool
from pydantic import AliasChoices, AnyUrl, BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator, validate_call
import uvicorn
import hydra
from omegaconf import DictConfig
import pandas as pd
from buttermilk import BM
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from promptflow.tracing import start_trace, trace
from urllib.parse import parse_qs

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import pubsub
import json
from buttermilk.agents.lc import LC
from buttermilk.llms import CHATMODELS
from buttermilk._core.runner_types import Job, RecordInfo, validate_uri_extract_text, validate_uri_or_b64
from buttermilk.runner import Job, MultiFlowOrchestrator
from buttermilk._core.runner_types import Result
from buttermilk.runner.helpers import group_and_filter_jobs
from buttermilk.utils.bq import TableWriter
from buttermilk.utils.save import upload_rows
from buttermilk.utils.utils import expand_dict, read_file
from buttermilk._core.log import logger

import httpx
from buttermilk.agents.agent import Agent, LC
from buttermilk.utils.validators import make_list_validator
from .runs import get_recent_runs


INPUT_SOURCE = "api"   
app = FastAPI()
 
# curl -X 'POST' 'http://127.0.0.1:8000/flow/judge' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": "haiku", "template":"judge", "formatting": "json_rules", "criteria": "criteria_ordinary", "text": "Republicans are arseholes."}'

# curl -X 'POST' 'http://127.0.0.1:8000/flow/judge' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": "haiku", "template":"judge", "formatting": "json_rules", "criteria": "criteria_ordinary", "video": "gs://dmrc-platforms/test/fyp/tiktok-imane-01.mp4"}'

# curl -X 'POST' 'http://127.0.0.1:8000/flow/judge' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": ["haiku", "gpt4o"], "template":"summarise_osb", "text": "gs://dmrc-platforms/data/osb/FB-UK2RUS24.md"}'




class FlowRequest(BaseModel):
    model: Optional[str|Sequence[str]] = None
    template: Optional[str|Sequence[str]] = None
    template_vars: Optional[dict|Sequence[dict]] = Field(default_factory=list)
    
    content: Optional[str] = Field(default=None, validation_alias=AliasChoices("content", "text", "body"))
    video: Optional[str] = None
    image: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=False, extra='allow',
        populate_by_name=True
    )
    _client: LC = PrivateAttr()
    _job: Job = PrivateAttr()

    _ensure_list = field_validator("model", "template", "template_vars", mode="before")(make_list_validator())
    
    @model_validator(mode='before')
    def preprocess_data(cls, values):
        # Check first if the values are coming as utf-8 encoded bytes
        try:
            values = values.decode('utf-8')
        except:
            pass
        try:
            # Might be HTML form data
            values = parse_qs(values)

            # Convert the values from lists to single values
            values = {key: value[0] for key, value in values.items()}
        except:
            pass

        try:
            # might be JSON
            values = json.loads(values)
        except:
            pass
    
        if not any([values.get('content'), values.get('uri'), values.get('text'), values.get('image'), values.get('video')]):
            raise ValueError("At least one of content, text, uri, video or image must be provided.")
        
        
        
        return values

    @field_validator('content', 'image', 'video', mode='before')
    def sanitize_strings(cls, v):
        if v:
            return v.strip()
        return v
    
    @model_validator(mode='after')
    def check_values(self) -> Self:
        for v in self.model:
            if v not in CHATMODELS:
                raise ValueError(f"Valid model must be provided. {v} is unknown.")
                
        # Add any additional vars into the template_vars dict.
        extras = []
        if self.template_vars and self.model_extra:
            # Template variables is a list of dicts that are run in combinations
            # When we add other variables, make sure to add them to each existing combination
            for existing_vars in self.template_vars:
                existing_vars.update(self.model_extra)
                extras.append(existing_vars)
        elif self.model_extra:
            self.template_vars = [self.model_extra]
        
        return self


async def run_flow(flow: Literal['judge','summarise'], request: Request, flow_request: Optional[FlowRequest] = '') -> AsyncGenerator[Job, None]:
    # Ensure at least one of the content, uri, or media variables are provided
    # And make sure that we can form Client and Job objects from the input data.
    content, image,video = await asyncio.gather(validate_uri_extract_text(flow_request.content), validate_uri_or_b64(flow_request.image), validate_uri_or_b64(flow_request.video))
    record = RecordInfo(content=content, image=image, video=video)
 
    init_vars = dict(template=flow_request.template, template_vars=flow_request.template_vars)
    agent_vars = dict(flow_obj=LC, agent=flow, save_params=bm.cfg.save, concurrent=bm.cfg.concurrent)
    run_vars = dict(model=flow_request.model)

    orchestrator = MultiFlowOrchestrator(n_runs=1, agent_vars=agent_vars, init_vars=init_vars, run_vars=run_vars)

    async for result in orchestrator.run_tasks(record=record, source=INPUT_SOURCE):
        yield result


def callback(message):
    results = None
    data = json.loads(message.data)
    task = data.pop("task")
    try:
        request = FlowRequest(**data)
        message.ack()
    except Exception as e:
        message.nack()
        logger.error(f"Error parsing Pub/Sub message: {e}")
    
    try:
        # Try to get existing loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop exists, create new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        should_close_loop = True
    else:
        should_close_loop = False

    try:
        # Run async operations in the loop
        bm.logger.info(f"Calling flow {task} for Pub/Sub job...")

        async def process_generator():
            results = []
            async for result in run_flow(flow=task, request=request, flow_request=request):
                results.append(result)
            return results
        
        results = loop.run_until_complete(process_generator())
        message.ack() 
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        message.nack()
    finally:
        if should_close_loop:
            loop.close()
    
    bm.logger.info(f"Passed on Pub/Sub job. Received {len(results)} results")

def start_pubsub_listener():
    subscriber = pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path(bm.cfg.gcp.project, 'flow-sub')
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening for messages on {subscription_path}...")

    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()


bm = BM()
templates = Jinja2Templates(directory="buttermilk/api/templates")

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

@app.api_route("/runs/", methods=["GET", "POST"])
async def get_runs_json(request: Request) -> Sequence[Job]:
    runs = get_recent_runs() 

    results = [ Job(**row) for _, row in runs.iterrows()]

    return results

@app.api_route("/html/runs/", methods=["GET", "POST"])
async def get_runs_html(request: Request) -> HTMLResponse:
    df = get_recent_runs() 

    data = group_and_filter_jobs(new_data=df, group=bm.cfg.data.runs.group, columns=bm.cfg.data.runs.columns, raise_on_error=False)

    rendered_result = templates.TemplateResponse(f"runs_html.html", {"request": request, "data": data})

    return HTMLResponse(rendered_result.body.decode('utf-8'), status_code=200)

@app.api_route("/flow/{flow}", methods=["GET", "POST"])
async def run_flow_json(flow: Literal['judge','summarise'], request: Request, flow_request: Optional[FlowRequest] = '') -> Sequence[Job]:
    # return StreamingResponse(run_flow(flow=flow, request=request, flow_request=flow_request), media_type="application/json")
    results = [ job async for job in run_flow(flow=flow, request=request, flow_request=flow_request)]
    return results

@app.api_route("/html/flow/{flow}", methods=["GET", "POST"])
@app.api_route("/html/flow", methods=["GET", "POST"])
async def run_route_html(request: Request, flow: str = '', flow_request: Optional[FlowRequest] = '') -> StreamingResponse:
    async def result_generator() -> AsyncGenerator[str, None]:
        bm.logger.info(f"Received request for flow {flow} and flow_request {flow_request}")
        try:
            async for data in run_flow(flow=flow, request=request, flow_request=flow_request):
                # Render the template with the response data
                rendered_result = templates.TemplateResponse(f"flow_html.html", {"request": request, "data": data})
                yield rendered_result.body.decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    return StreamingResponse(result_generator(), media_type="text/html")



# Set up CORS
origins = [
    "http://localhost:5000",# Frontend running on localhost:5000
    "http://127.0.0.1:5000",
    "http://127.0.0.1:8080",# Frontend running on localhost:8080
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
        bm.logger.debug(f"CORS check for {origin}")
    
    response = await call_next(request)
    return response

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    global bm, logger, app
    bm = BM(cfg=cfg)
    
    logger = bm.logger
    start_trace(resource_attributes={"run_id": bm._run_metadata.run_id}, collection="flow_api", job="pubsub prompter")

    listener_thread = threading.Thread(target=start_pubsub_listener)
    listener_thread.start()
        
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # writer = TableWriter(**cfg.save)


if __name__ == "__main__":
    main()
    
