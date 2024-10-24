import asyncio
import json
import os
import sys
from pathlib import Path

import threading
from cloudpathlib import AnyPath
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
from buttermilk.utils.utils import read_file

app = FastAPI()
bm = None
logger = None

# Initialize a semaphore with a concurrency limit
semaphore = asyncio.Semaphore(5)

@app.post("/format")
async def process_job(job: dict):
    async with semaphore:
        basename = None
        if content := job.get("content"):
            pass
        elif filename := job.get("filename"):
            content = read_file(filename)
            basename = AnyPath(filename).stem

        flow = LC(model=job.get("model", "gpt4o"), template=job.get("template", "format_osb"), )

        response = await flow.call_async(content=content)
        bm.save(data=response, basename=basename, ext="json")
        response.update({"status": "Job processed successfully"})
        return response



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
