import asyncio
import contextlib
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.websockets import WebSocketState

from buttermilk._core import BM, logger
from buttermilk._core.config import FatalError
from buttermilk._core.context import session_id_var
from buttermilk._core.types import RunRequest
from buttermilk.runner.flowrunner import FlowRunner

from .lazy_routes import LazyRouteManager, create_core_router
from .routes import flow_data_router
from .mcp import mcp_router
from .mcp_agents import agent_mcp_router

# Define the base directory for the FastAPI app
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
INPUT_SOURCE = "api"


def create_app(bm: BM, flows: FlowRunner) -> FastAPI:
    """Create and configure the FastAPI application."""
    logger.info("Starting create_app function...")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan event handler for startup and shutdown events.
        """
        try:
            # Complete BM initialization in the FastAPI event loop
            if hasattr(app.state, "bm"):
                await app.state.bm._background_init()
            yield
        except Exception as e:
            logger.error(f"Failed to start job worker: {e}")
        finally:
            # Shutdown event
            if hasattr(app.state, "job_worker"):
                await app.state.job_worker.stop()
                logger.info("Stopped job worker")

            # Clean up FlowRunner sessions
            if hasattr(app.state, "flow_runner"):
                try:
                    await app.state.flow_runner.cleanup()
                    logger.info("Cleaned up FlowRunner sessions")
                except Exception as e:
                    logger.error(f"Error cleaning up FlowRunner: {e}")

    # Create the FastAPI app with the lifespan
    app = FastAPI(lifespan=lifespan)

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

    # Example usage:
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/{flow_name}' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"q": "Your text content here"}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/{flow_name}' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"text": "Your text content here"}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/{flow_name}' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"uri": "https://example.com/article"}'
    # curl -X 'POST' 'http://127.0.0.1:8000/flow/{flow_name}' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"record_id": "your_record_id"}'

    logger.info("Defining exception handler.")

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    logger.info("Setting up lazy route management.")

    # Initialize lazy route manager for Phase 2 optimization
    lazy_manager = LazyRouteManager(app)

    # Register core routes immediately (essential functionality)
    core_router = create_core_router()
    app.include_router(core_router)
    lazy_manager.register_core_routes()

    logger.info("Core routes registered, deferring heavy routes.")

    # Set up CORS

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

    # WebSocket endpoint - essential for frontend terminal functionality
    @flow_data_router.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for client communication with the WebUIAgent.

        Args:
            websocket: WebSocket connection
            session_id: Unique identifier for this client session

        """
        # Accept the WebSocket connection
        if websocket.client_state == WebSocketState.CONNECTING:
            logger.debug(f"Accepting WebSocket connection for session {session_id}")
            await websocket.accept()
        flow_runner: FlowRunner = websocket.app.state.flow_runner
        if not (session := await flow_runner.get_websocket_session_async(session_id=session_id, websocket=websocket)):
            logger.error(f"Session {session_id} not found.")
            await websocket.close()
            raise HTTPException(status_code=404, detail="Session not found")

        task = None
        # Listen for messages from the client

        token = session_id_var.set(session_id)
        async for run_request in session.monitor_ui():
            try:
                await asyncio.sleep(0.1)
                # This loop internally feeds the groupchat with messages from the client.
                # The only message we receive is a run_request -- which we then
                # use to create a new flow.
                task = asyncio.create_task(flow_runner.run_flow(
                    run_request=run_request,
                    wait_for_completion=False,
                ))

            except WebSocketDisconnect:
                logger.info(f"Client {session_id} disconnected.")
                break
            except Exception as e:
                msg = f"Error receiving/processing client message for {session_id}: {e}"
                raise FatalError(msg) from e
            finally:
                session_id_var.reset(token)

        if task:
            await task

        # Clean up the session when WebSocket disconnects
        try:
            if hasattr(flow_runner, 'session_manager'):
                success = await flow_runner.session_manager.cleanup_session(session_id)
                if success:
                    logger.info(f"Cleaned up session {session_id} after WebSocket disconnect")
                else:
                    logger.debug(f"Session {session_id} not found during cleanup (may have already been cleaned up)")
        except Exception as e:
            logger.warning(f"Error cleaning up session {session_id}: {e}")

        with contextlib.suppress(Exception):
            await websocket.close()

    # Session management routes - essential for frontend functionality
    @app.get("/api/session")
    async def create_session():
        """Generates a unique session ID for new web clients.

        Returns:
            Dict with new session ID
        """
        return {"session_id": str(uuid.uuid4())}

    # Session management endpoints
    @app.get("/api/session/{session_id}/status")
    async def get_session_status(session_id: str, request: Request):
        """Get the status of a specific session.
        
        Returns:
            Dict with session status information
        """
        flow_runner: FlowRunner = request.app.state.flow_runner

        if not hasattr(flow_runner, 'session_manager'):
            raise HTTPException(status_code=500, detail="Session manager not available")

        if session_id in flow_runner.session_manager.sessions:
            session = flow_runner.session_manager.sessions[session_id]
            return {
                "session_id": session_id,
                "status": session.status,
                "flow_name": session.flow_name,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "is_expired": session.is_expired()
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    @app.delete("/api/session/{session_id}")
    async def cleanup_session(session_id: str, request: Request):
        """Manually clean up a specific session.
        
        Returns:
            Dict confirming cleanup
        """
        flow_runner: FlowRunner = request.app.state.flow_runner

        if not hasattr(flow_runner, 'session_manager'):
            raise HTTPException(status_code=500, detail="Session manager not available")

        success = await flow_runner.session_manager.cleanup_session(session_id)
        if success:
            return {"message": f"Session {session_id} cleaned up successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    @app.get("/api/sessions")
    async def list_sessions(request: Request):
        """List all active sessions.
        
        Returns:
            Dict with list of session information
        """
        flow_runner: FlowRunner = request.app.state.flow_runner

        if not hasattr(flow_runner, 'session_manager'):
            return {"sessions": [], "total": 0}

        sessions_info = []
        for session_id, session in flow_runner.session_manager.sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "status": session.status,
                "flow_name": session.flow_name,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "is_expired": session.is_expired()
            })
        return {"sessions": sessions_info, "total": len(sessions_info)}

    # --- Defer heavy routes for Phase 2 optimization ---
    # Set up templates
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Defer heavy routers until first request
    lazy_manager.defer_router(flow_data_router, prefix="")
    lazy_manager.defer_router(mcp_router, prefix="")
    lazy_manager.defer_router(agent_mcp_router, prefix="")
    lazy_manager.create_lazy_middleware()

    logger.info("Heavy routes deferred - will load on first request")

    # Set up static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app
