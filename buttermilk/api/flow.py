import asyncio
import itertools
import json
import os
import sys
from pathlib import Path

import threading
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Self, Sequence
from cloudpathlib import AnyPath
from pydantic import AliasChoices, AnyUrl, BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator, validate_call
import uvicorn
import hydra
from omegaconf import DictConfig
import pandas as pd
from buttermilk import BM
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from promptflow.tracing import start_trace, trace
from urllib.parse import parse_qs

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import pubsub
import json
from buttermilk.lc import LC
from buttermilk.llms import CHATMODELS
from buttermilk.runner._runner_types import Job, RecordInfo, validate_uri_extract_text, validate_uri_or_b64
from buttermilk.runner import Job
from buttermilk.runner._runner_types import Result
from buttermilk.utils.bq import TableWriter
from buttermilk.utils.save import upload_rows
from buttermilk.utils.utils import expand_dict, read_file

import httpx
from buttermilk.flows.agent import Agent
from buttermilk.utils.validators import make_list_validator
from .runs import get_recent_runs

# curl -X 'POST' 'http://127.0.0.1:8000/flow/judge' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": "haiku", "template":"judge", "formatting": "json_rules", "criteria": "criteria_ordinary", "text": "Republicans are arseholes."}'

# curl -X 'POST' 'http://127.0.0.1:8000/flow/judge' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": "haiku", "template":"judge", "formatting": "json_rules", "criteria": "criteria_ordinary", "video": "gs://dmrc-platforms/test/fyp/tiktok-imane-01.mp4"}'

# curl -X 'POST' 'http://127.0.0.1:8000/flow/judge' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": ["haiku", "gpt4o"], "template":"summarise_osb", "text": "gs://dmrc-platforms/data/osb/FB-UK2RUS24.md"}'
class FlowProcessor(Agent):
    async def process_job(self, job: Job) -> Job:
        response = await self.client.call_async(record=job.record, params=job.parameters)
        job.outputs = Result(**response)
        job.agent_info = self.agent_info
        return job

class MultiFlowOrchestrator(BaseModel):
    agents: Optional[Sequence[Agent]] = Field(default_factory=list)
    n_runs: int = 1
    steps: Optional[Sequence] = Field(default_factory=list)
    agent_vars: Optional[dict] = Field(default_factory=dict)
    init_vars: Optional[dict] = Field(default_factory=dict)
    run_vars: Optional[dict] = Field(default_factory=dict)

    async def run_tasks(self, record: RecordInfo, source: str) -> AsyncGenerator[Job, None]:
        init_combinations = expand_dict(self.init_vars)
        run_combinations = expand_dict(self.run_vars)

        self.agents = [FlowProcessor(**vars, **self.agent_vars) for vars in init_combinations]

        # Multiply by n
        permutations = run_combinations * self.n_runs
        
        # For each agent, create tasks for each job
        workers = []
        for agent in self.agents:
            for vars in permutations:
                job = Job(record=record, source=INPUT_SOURCE, parameters=vars)
                workers.append(agent.run(job))
        
        # Process tasks as they complete
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(worker) for worker in workers]
            
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    yield result
                except Exception as e:
                    logger.error(f"Worker failed with error: {e}")
                    continue
        
        # All workers are now complete
        return


INPUT_SOURCE = "api"    
bm = None
logger = None
app = FastAPI()

class FlowRequest(BaseModel):
    model: Optional[str|Sequence[str]] = None
    template: Optional[str|Sequence[str]] = None
    template_vars: Optional[dict|Sequence[dict]] = Field(default_factory=list)
    
    content: Optional[str] = Field(default=None, validation_alias=AliasChoices("content", "text", "body"))
    video: Optional[str] = None
    image: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=False, extra='allow')

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
    
        # Add any additional vars into the template_vars dict.
        kwargs = { k: values.pop(k) for k in values.copy().keys() if k not in cls.model_fields.keys() }
        if 'template_vars' not in values:
            values['template_vars'] = {}
        values['template_vars'].update(**kwargs)
        
        if not any([values.get('content'), values.get('image'), values.get('video')]):
            raise ValueError("At least one of content, video or image must be provided.")
        
        return values

    @field_validator('content', 'image', 'video', mode='before')
    def sanitize_strings(cls, v):
        if v:
            return v.strip()
        return v
    
    @field_validator('model', mode='before')
    def check_model(cls, value: Optional[Sequence[str]]) ->  Optional[Sequence[str]]:
        for v in value:
            if v not in CHATMODELS:
                raise ValueError(f"Valid model must be provided. {v} is unknown.")
        return value


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
    data = json.loads(message.data)
    message.ack()
    task = data.pop("task")

    # Call your FastAPI endpoint to process the job
    # You can use requests or httpx to make the HTTP call
    import requests
    response = requests.post(f"http://localhost:8000/{task}", json=data)
    logger.info(f"Passed on Pub/Sub job. Received: {response.json()}")

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
logger = None
templates = Jinja2Templates(directory="buttermilk/api/templates")

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

@app.api_route("/runs/", methods=["GET", "POST"])
async def get_runs(request: Request) -> Sequence[Job]:
    runs = get_recent_runs() 
    return runs

@app.api_route("/flow/{flow}", methods=["GET", "POST"])
async def run_flow_json(flow: Literal['judge','summarise'], request: Request, flow_request: Optional[FlowRequest] = '') -> Sequence[Job]:
    # return StreamingResponse(run_flow(flow=flow, request=request, flow_request=flow_request), media_type="application/json")
    results = [ job async for job in run_flow(flow=flow, request=request, flow_request=flow_request)]
    return results

@app.api_route("/html/{route}/{flow}", methods=["GET", "POST"])
@app.api_route("/html/{route}", methods=["GET", "POST"])
async def run_route_html(route: str, request: Request, flow: str, flow_request: Optional[FlowRequest] = '') -> HTMLResponse:
    # Route the request to the "/{route}/{flow}" function with original args
    async def result_generator() -> AsyncGenerator[Job, None]:
        async for data in run_flow(flow=flow, request=request, flow_request=flow_request):
            # Render the template with the response data
            result = templates.TemplateResponse(f"{route}_html.html", {"request": request, "data": data})
            yield result

    return StreamingResponse(result_generator(), media_type="application/html")

def run_app(cfg: DictConfig) -> None:
    global bm, logger, app 
    bm = BM(cfg=cfg)
    logger = bm.logger

    start_trace(resource_attributes={"run_id": bm._run_metadata.run_id}, collection="flow_api", job="pubsub prompter")

    listener_thread = threading.Thread(target=start_pubsub_listener)
    listener_thread.start()
        
    # writer = TableWriter(**cfg.save)

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
    uvicorn.run(app, host="0.0.0.0", port=8000)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_app(cfg)

if __name__ == "__main__":
    main()
