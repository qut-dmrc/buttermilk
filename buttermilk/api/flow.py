import asyncio
import json
import uuid
from contextlib import asynccontextmanager

import pydantic
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from buttermilk._core.contract import ErrorEvent
from buttermilk._core.types import RunRequest
from buttermilk.api.job_queue import JobQueueClient
from buttermilk.bm import BM, logger
from buttermilk.runner.flowrunner import FlowRunner
from buttermilk.web.activity_tracker import get_instance as get_activity_tracker

INPUT_SOURCE = "api"


# Middleware to track API activity
class ActivityTrackerMiddleware(BaseHTTPMiddleware):
    """Middleware that tracks API requests for the ActivityTracker."""

    async def dispatch(self, request: Request, call_next):
        # Record the API request in the activity tracker
        activity_tracker = get_activity_tracker()
        activity_tracker.record_api_request()

        # Process the request as usual
        response = await call_next(request)
        return response


def create_app(bm: BM, flows: FlowRunner) -> FastAPI:
    """Create and configure the FastAPI application."""
    logger.info("Starting create_app function...")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan event handler for startup and shutdown events.
        """
        try:
            # Startup event
            worker = JobQueueClient(
                flow_runner=flows,
                max_concurrent_jobs=1,
            )
            app.state.job_worker = worker
            logger.info("Started job worker in FastAPI application")
            task = asyncio.create_task(worker.pull_tasks())
            yield
        except Exception as e:
            logger.error(f"Failed to start job worker: {e}")
        finally:
            # Shutdown event
            if hasattr(app.state, "job_worker"):
                await app.state.job_worker.stop()
                logger.info("Stopped job worker")

    # Create the FastAPI app with the lifespan
    app = FastAPI(lifespan=lifespan)

    # Add the activity tracker middleware
    app.add_middleware(ActivityTrackerMiddleware)

    logger.info("FastAPI() instance created.")

    # Set up state
    app.state.bm = bm
    app.state.flow_runner = flows

    # Initialize batch runner
    logger.info("App state configured.")

    # Add batch router
#    batch_router = create_batch_router(app.state.batch_runner)
    # app.include_router(batch_router, prefix="/api")
    # logger.info("Batch router added.")

    # curl -X 'POST' 'http://127.0.0.1:8000/flow/simple' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"q": "Democrats are arseholes."}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/simple' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"text": "Democrats are arseholes."}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/trans' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"uri": "https://www.city-journal.org/article/what-are-we-doing-to-children"}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/osb' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"q": "Is it still hate speech if the targeted group is not explicitly named?"}'

    # curl -X 'POST' 'http://127.0.0.1:8000/flow/hate' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": ["haiku", "gpt4o"], "criteria": "criteria_ordinary", "video": "gs://dmrc-platforms/test/fyp/tiktok-imane-01.mp4"}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/hate' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"uri": "https://upload.wikimedia.org/wikipedia/en/b/b9/MagrittePipe.jpg"}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/judge' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"model": ["haiku", "gpt4o"], "template":"summarise_osb", "text": "gs://dmrc-platforms/data/osb/FB-UK2RUS24.md"}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/trans' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"record_id": "betoota_snape_trans"}'

    logger.info("Defining exception handler.")

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    logger.info("Defining API routes.")

    @app.api_route("/flow/{flow_name}", methods=["GET", "POST"])
    async def run_flow_json(
        flow_name: str,
        request: Request,
        run_request: RunRequest | None = None,
    ) -> StreamingResponse:
        """Run a flow with provided inputs."""
        # Access state via request.app.state
        if not hasattr(request.app.state.flow_runner, "flows") or flow_name not in request.app.state.flow_runner.flows:
            raise HTTPException(status_code=404, detail="Flow configuration not found or flow name invalid")

        # Use stream method with the flow runner
        return StreamingResponse(
            request.app.state.flow_runner.stream(run_request),
            media_type="application/json",
        )

    # Note: The following route is kept for backward compatibility but may be removed in future versions
    @app.api_route("/html/flow/{flow}", methods=["GET", "POST"])
    @app.api_route("/html/flow", methods=["GET", "POST"])
    async def run_route_html(
        request: Request,
        flow: str = "",
        flow_request: RunRequest | None = None,
    ) -> JSONResponse:
        """This endpoint is deprecated - please use the main API or WebSocket endpoints"""
        return JSONResponse(
            content={"message": "HTML flow rendering is deprecated. Please use the main API or WebSocket endpoints."},
            status_code=200,
        )

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

    @app.websocket("/ws/{session_id}")
    async def agent_websocket(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for client communication with the WebUIAgent.
        
        Args:
            websocket: WebSocket connection
            session_id: Unique identifier for this client session

        """
        # Accept the connection first
        await websocket.accept()
        client_listener_task = None
        try:
            # Start the flow, passing in our websocket and session_id
            # Wait for client message
            while True:
                client_message = await websocket.receive_json()
                try:
                    run_request = RunRequest.model_validate(client_message)
                    run_request.client_callback = websocket
                    run_request.session_id = session_id
                    break
                except (pydantic.ValidationError, json.JSONDecodeError):
                    await websocket.send_json(ErrorEvent(source="fastapi flow websocket", content="Send a valid RunRequest to start."))

            agent_callback = await app.state.flow_runner.run_flow(run_request)

            async def listen_client():
                """Task to listen for incoming messages from the client."""
                while True:
                    try:
                        incoming_data = await websocket.receive_json()
                        # Forward data to the running flow via the handler
                        await agent_callback(incoming_data)
                    except WebSocketDisconnect:
                        logger.info(f"Client {session_id} disconnected.")
                        # Optionally signal the flow task to stop if needed
                        # if flow_task_handle:
                        #     flow_task_handle.cancel()
                        break  # Exit loop on disconnect
                    except Exception as e:
                        logger.error(f"Error receiving/processing client message for {session_id}: {e}")
                        # Decide if you want to break or continue
                        # break # Exit loop on other errors

            client_listener_task = asyncio.create_task(listen_client())

            await client_listener_task

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Unexpected error: {e!s}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "content": f"Unexpected error: {e!s}",
                    "source": "system",
                })
            except:
                pass
        finally:
            if client_listener_task and not client_listener_task.done():
                client_listener_task.cancel()

    # Helper route to generate session IDs for clients
    @app.get("/api/session")
    async def create_session():
        """Generates a unique session ID for new web clients.
        
        Returns:
            Dict with new session ID

        """
        return {"session_id": str(uuid.uuid4())}

    # --- Import Shiny App object ---
    from buttermilk.web.shiny import get_shiny_app

    # --- Mount the Shiny App ---
    shiny_app_asgi = get_shiny_app(flows=flows)
    app.mount("/ui", shiny_app_asgi, name="shiny_app")

    logger.info("Importing Dashboard app")

    # --- Import Dashboard App object ---
    from buttermilk.web.fastapi_frontend.app import create_dashboard_app

    # --- Mount the Dashboard App ---
    logger.info("Getting Dashboard app.")
    dashboard_app = create_dashboard_app(flows=flows)
    logger.info("Mounting Dashboard app.")
    app.mount("/dash", dashboard_app, name="dashboard_app")

    return app
