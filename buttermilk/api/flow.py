import asyncio
import json
import os
import sys
from pathlib import Path

import threading
from typing import Literal, Optional, Self, Sequence
from cloudpathlib import AnyPath
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator, validator
import uvicorn
import hydra
from omegaconf import DictConfig
import pandas as pd
from buttermilk import BM
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
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
from buttermilk.runner._runner_types import Job
from buttermilk.runner import Job
from buttermilk.runner._runner_types import Result
from buttermilk.utils.bq import TableWriter
from buttermilk.utils.save import upload_rows
from buttermilk.utils.utils import read_file
import httpx
from buttermilk.flows.agent import Agent
from .runs import get_recent_runs

# curl -X 'POST' 'http://127.0.0.1:8000/flow/judge' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": "haiku", "template":"judge", "formatting": "json_rules", "criteria": "criteria_ordinary", "text": "Republicans are arseholes."}'
class FlowProcessor(Agent):
    client: Optional[LC] = LC
    
    async def process_job(self, job: Job) -> Job:
        response = await self.client.call_async(**job.inputs)
        job.outputs = Result(**response)
        job.agent_info = self.agent_info
        return job

INPUT_SOURCE = "api"    
bm = None
logger = None
app = FastAPI()

class FlowRequest(BaseModel):
    model: Optional[str] = None
    template: Optional[str] = None
    template_vars: Optional[dict] = Field(default_factory=dict)
    
    text: Optional[str] = None
    uri: Optional[AnyUrl] = None
    media_b64: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=False, extra='allow')

    _client: LC = PrivateAttr()
    _job: Job = PrivateAttr()

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
        
        if not any([values.get('text'), values.get('uri'), values.get('media_b64')]):
            raise ValueError("At least one of content, uri, or media_b64 must be provided.")
        
        return values

    @field_validator('text', 'uri', 'media_b64', mode='before')
    def sanitize_strings(cls, v):
        if v:
            return v.strip()
        return v
    
    @field_validator('model', mode='before')
    def check_model(cls, value: Optional[str]) -> str:
        if value not in CHATMODELS:
            raise ValueError("Valid model must be provided")
        return value

    # Ensure at least one of the content, uri, or media variables are provided
    # And make sure that we can form Client and Job objects from the input data.
    @model_validator(mode='after')
    def create_models(self) -> Self:
        self._client = LC(model=self.model, template=self.template, template_vars=self.template_vars)

        inputs = dict(text=self.text, uri=self.uri, media_b64=self.media_b64)
        inputs = {k:v for k,v in inputs.items() if v}
        self._job = Job(inputs=inputs, source=INPUT_SOURCE)

        return self

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

@app.api_route("/flow/{flow}", methods=["GET", "POST"])
async def run_flow(flow: str, request: Request, flow_request: Optional[FlowRequest] = '') -> Job:
    agent = FlowProcessor(client=flow_request._client, agent=flow, save_params=bm.cfg.save, concurrent=bm.cfg.concurrent)
    result = await agent.run(job=flow_request._job)
    # writer.append_rows(rows=rows)
    return result


@app.api_route("/runs/", methods=["GET", "POST"])
async def get_runs(request: Request) -> Sequence[Job]:
    runs = get_recent_runs()
    return runs

@app.api_route("/html/{route}/{flow}", methods=["GET", "POST"])
@app.api_route("/html/{route}", methods=["GET", "POST"])
async def run_route_html(route: str, request: Request, flow: Optional[str] = '') -> HTMLResponse:
    # Route the request to "/{route}/{flow}" with original args
    async with httpx.AsyncClient() as client:
        if request.method == "GET":
            response = await client.get(f"http://localhost:8000/{route}/{flow}", params=request.query_params)
        elif request.method == "POST":
            response = await client.post(f"http://localhost:8000/{route}/{flow}", json=await request.json())
    # Process the response
    response_data = response.json()
    
    # Render the template with the response data
    result = templates.TemplateResponse(f"{route}_html.html", {"request": request, "data": response})
    return result

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
