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
from promptflow.tracing import start_trace
from pydantic import BaseModel

from buttermilk import BM, logger
from buttermilk._core.runner_types import Job
from buttermilk.api.stream import FlowRequest, flow_stream
from buttermilk.runner.flow import Flow
from buttermilk.utils.utils import load_json_flexi

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
# gcloud pubsub topics publish TOPIC_ID --message='{"task": "summarise_osb", "uri": "gs://dmrc-platforms/data/osb/FB-515JVE4X.md", "record_id": "FB-515JVE4X"}


def callback(message):
    results = None
    try:
        data = load_json_flexi(message.data)
        task = data.pop("task")
        request = FlowRequest(**data)
        message.ack()
    except Exception as e:
        message.nack()
        logger.error(f"Error parsing Pub/Sub message: {e}")
        return

    try:
        logger.info(f"Calling flow {task} for Pub/Sub job...")

        async def process_generator():
            results = []
            async for result in flow_stream(flows[task], request):
                results.append(result)
            return results

        results = asyncio.run(
            process_generator(),
        )
        message.ack()

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        message.nack()

    logger.info("Completed Pub/Sub job.")


def start_pubsub_listener():
    # publisher = pubsub.PublisherClient()
    subscriber = pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path(
        bm.cfg.pubsub.project,
        bm.cfg.pubsub.subscription,
    )
    topic_path = subscriber.topic_path(bm.cfg.pubsub.project, bm.cfg.pubsub.topic)

    # if "dead_letter_topic_id" in bm.cfg.pubsub:
    #     dead_letter_topic_path = publisher.topic_path(
    #         bm.cfg.pubsub.project,
    #         bm.cfg.pubsub.dead_letter_topic,
    #     )
    #     logger.info(
    #         "Pub/Sub forwarding failed messages to: {dead_letter_topic_path} after {bm.cfg.pubsub.max_retries} retries.",
    #     )
    #     dead_letter_policy = {
    #         "dead_letter_topic": dead_letter_topic_path,
    #         "max_delivery_attempts": bm.cfg.pubsub.max_retries,
    #     }
    # else:
    #     dead_letter_policy = None

    # # try to create the subscription if necessary
    # try:
    #     with subscriber:
    #         request = {
    #             "name": subscription_path,
    #             "topic": topic_path,
    #             "dead_letter_policy": dead_letter_policy,
    #         }
    #         subscription = subscriber.create_subscription(request)
    # except Exception as e:
    #     logger.error(
    #         f"Unable to create pub/sub subscription {subscription_path}: {e}, {e.args=}",
    #     )

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    logger.info(f"Listening for messages on {subscription_path} topic {topic_path}...")

    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()


bm = None
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


@app.api_route("/flow/{flow}", methods=["GET", "POST"])
async def run_flow_json(
    flow: Literal[
        "hate",
        "simple",
        "trans",
        "osb",
        "osbfulltext",
        "summarise_osb",
        "test",
        "describer",
    ],
    request: Request,
    flow_request: FlowRequest | None = "",
) -> StreamingResponse:
    if flow not in flows:
        raise HTTPException(status_code=403, detail="Flow not valid")

    return StreamingResponse(
        flow_stream(flows[flow], flow_request),
        media_type="application/json",
    )


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


class _CFG(BaseModel):
    bm: BM
    save: Mapping
    flows: dict[str, Flow]


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: _CFG):
    global bm, logger, app, flows

    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)

    bm = objs.bm
    flows = objs.flows

    logger = logger

    listener_thread = threading.Thread(target=start_pubsub_listener)
    listener_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
    # writer = TableWriter(**cfg.save)


if __name__ == "__main__":
    main()
