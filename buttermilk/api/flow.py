import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from buttermilk._core.types import RunRequest
from buttermilk.api.job_queue import JobQueueClient
from buttermilk.api.services.websocket_service import WebSocketManager
from buttermilk.bm import BM, logger
from buttermilk.runner.flowrunner import FlowRunner
from buttermilk.web.activity_tracker import get_instance as get_activity_tracker

from .routes import flow_data_router

# Define the base directory for the FastAPI app
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
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
            # # Startup event
            # worker = JobQueueClient(
            #     max_concurrent_jobs=1,
            # )
            # app.state.job_worker = worker
            # logger.info("Started job worker in FastAPI application")
            # task = asyncio.create_task(worker.pull_tasks())
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
    app.state.websocket_manager = WebSocketManager()

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

    # Set up CORS
    origins = [
        "http://localhost:5000",  # Frontend running on localhost:5000
        "http://127.0.0.1:5000",
        "http://127.0.0.1:5173",  # Frontend running on localhost:5173
        "http://localhost:5173",
        "http://localhost:8000",  # Allow requests from localhost:8000
        "http://127.0.0.1:8000",
        "http://localhost",
        "http://127.0.0.1",
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

    @flow_data_router.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for client communication with the WebUIAgent.
        
        Args:
            websocket: WebSocket connection
            session_id: Unique identifier for this client session

        """
        manager: WebSocketManager = websocket.app.state.websocket_manager
        flow_runner: FlowRunner = websocket.app.state.flow_runner

        # Accept the connection first
        await manager.connect(websocket, session_id)

        try:
            # Listen for messages from the client
            while True:
                data = await websocket.receive_json()
                await manager.process_message(session_id, data, flow_runner)

        except WebSocketDisconnect:
            logger.info(f"Client {session_id} disconnected.")
        except Exception as e:
            logger.error(f"Error receiving/processing client message for {session_id}: {e}")

            try:
                await websocket.send_json({
                    "type": "error",
                    "content": f"Unexpected error: {e!s}",
                    "source": "system",
                })
            except:
                pass
        finally:
            manager.disconnect(session_id)

    # Helper route to generate session IDs for clients
    @app.get("/api/session")
    async def create_session():
        """Generates a unique session ID for new web clients.
        
        Returns:
            Dict with new session ID

        """
        return {"session_id": str(uuid.uuid4())}

    # --- Add API data routes ---
    # Set up templates
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.include_router(flow_data_router)

    # Set up static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app
