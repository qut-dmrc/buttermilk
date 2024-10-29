import asyncio
import json
import os
import sys
from pathlib import Path

import threading
from typing import Literal, Optional, Self
from cloudpathlib import AnyPath
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
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
from buttermilk.flows.agent import Agent

# curl -X 'POST' 'http://127.0.0.1:8000/flow' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"source": "api", "inputs":{"content": "Hello, how are you?"}}'
class FlowProcessor(Agent):
    client: Optional[LC] = LC
    
    async def _process(self, *, job: Job) -> Job:
        response = await self.client.call_async(**job.inputs)
        job.outputs = Result(**response)
        job.agent_info = self.agent_info
        return job

INPUT_SOURCE = "api"    
class FlowRequest(BaseModel):
    model: Optional[str] = None
    template: Optional[str] = None
    template_vars: Optional[dict] = Field(default_factory=dict)
    
    content: Optional[str] = None
    uri: Optional[str] = None
    media_b64: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=False, extra='allow')

    _client: LC = PrivateAttr()
    _job: Job = PrivateAttr()

    @model_validator(mode='before')
    def preprocess_data(cls, values):
        # Add any additional vars into the template_vars dict.
        kwargs = { k: values.pop(k) for k in values.copy().keys() if k not in cls.model_fields.keys() }
        values['template_vars'].update(**kwargs)

        if not any([values.get('content'), values.get('uri'), values.get('media_b64')]):
            raise ValueError("At least one of content, uri, or media_b64 must be provided.")
        return values

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

        inputs = dict(content=self.content, uri=self.uri, media_b64=self.media_b64)
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


bm = None
logger = None

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    global bm, logger
    bm = BM(cfg=cfg)
    logger = bm.logger
    start_trace(resource_attributes={"run_id": bm._run_metadata.run_id}, collection="flow_api", job="pubsub prompter")

    listener_thread = threading.Thread(target=start_pubsub_listener)
    listener_thread.start()
        
    # writer = TableWriter(**cfg.save)
    app = FastAPI()

    @app.post("/flow/{flow}")
    async def process_job(flow: str, request: FlowRequest):
        agent = FlowProcessor(client=request._client, agent=flow, **cfg.save, concurrent=cfg.concurrent)
        result = await agent._process(job=request._job)
        # writer.append_rows(rows=rows)
        return result

    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
