import asyncio
import json
import os
import sys
from pathlib import Path

import threading
from typing import Optional, Self
from cloudpathlib import AnyPath
from pydantic import BaseModel, model_validator
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
# from buttermilk.runner import Job
# from buttermilk.runner._runner_types import Result
# from buttermilk.utils.utils import read_file
# from buttermilk.flows.agent import Agent


# class FlowProcessor(Agent):
#     _client: Optional[LC] = None

#     @model_validator(mode='after')
#     def init(self) -> Self:
#         self._client = self.LC(**self.cfg.init_vars)

#         return self
     
#     async def process(self, *, job: Job) -> Job:
#         response = await self._client.call_async(**job.inputs)
#         job.outputs = Result(**response)
#         job.step_info = self.step_info
#         return job

class TestJob(BaseModel):
    input: int
    output: Optional[int] = None
class TestAgent(BaseModel):
    async def process(self, *, job: TestJob) -> TestJob:
        job.output = 2 * job.input
        return job


## TEST
## curl -X 'POST' 'http://127.0.0.1:8000/flow' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"input":4}'

app = FastAPI()
bm = None
logger = None
agent = TestAgent()

# Initialize a semaphore with a concurrency limit
semaphore = asyncio.Semaphore(5)

@app.post("/flow")
async def process_job(job: TestJob):
    result = await agent.process(job=job)
    return result

def callback(message):
    job = json.loads(message.data)
    message.ack()
    task = job.pop("task")

    # Call your FastAPI endpoint to process the job
    # You can use requests or httpx to make the HTTP call
    import requests
    response = requests.post(f"http://localhost:8000/{task}", json=job)
    print(response.json())

def start_pubsub_listener():

    subscriber = pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path(bm.cfg.gcp.project, 'flow-sub')
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening for messages on {subscription_path}...")

    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    global bm, logger
    bm = BM(cfg=cfg)
    logger = bm.logger
    start_trace(resource_attributes={"run_id": bm.run_id}, collection="flow_api", job="pubsub prompter")

    listener_thread = threading.Thread(target=start_pubsub_listener)
    listener_thread.start()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
