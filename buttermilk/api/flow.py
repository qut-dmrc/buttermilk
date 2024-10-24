import json
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
import pandas as pd
from buttermilk import BM
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


WEBDATA = Path("data")

from google.cloud import pubsub
import json

app = FastAPI()

@app.post("/process-job/")
async def process_job(job: dict):
    # Add your job processing logic here
    print(f"Processing job: {job}")
    return {"status": "Job processed successfully"}

def callback(message):
    job = json.loads(message.data)
    # Call your FastAPI endpoint to process the job
    # You can use requests or httpx to make the HTTP call
    import requests
    response = requests.post("http://localhost:8000/process-job/", json=job)
    print(response.json())
    message.ack()

def start_pubsub_listener():
    subscriber = pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path('your-project-id', 'your-subscription-id')
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening for messages on {subscription_path}...")

    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()

if __name__ == "__main__":
    import threading
    listener_thread = threading.Thread(target=start_pubsub_listener)
    listener_thread.start()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


app = FastAPI()
bm = None
logger = None
webdata = WebData(data_path=WEBDATA)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index(request: Request):
    title = "ChatGPT vs The Oversight Board"
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request, "webdata": webdata, "title": title}
    )


@app.post("/moderate")
async def moderate(request: Request):
    pass


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    global bm, logger
    bm = BM(cfg=cfg)
    logger = bm.logger

if __name__ == "__main__":
    main()
