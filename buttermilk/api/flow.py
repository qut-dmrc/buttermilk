import asyncio
import contextlib
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.websockets import WebSocketState

from buttermilk._core import BM, logger
from buttermilk._core.config import FatalError
from buttermilk._core.context import session_id_var
from buttermilk.runner.flowrunner import FlowRunner

from .lazy_routes import LazyRouteManager, create_core_router
from .monitoring import monitoring_router
from .routes import flow_data_router

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

            # API functions might take a few more seconds
            asyncio.get_event_loop().slow_callback_duration = 2

            # Complete BM initialization in the FastAPI event loop
            if hasattr(app.state, "bm") and hasattr(app.state.bm, "_background_init"):
                await app.state.bm._background_init()

            # Initialize and start monitoring infrastructure
            from buttermilk.monitoring import get_observability_manager
            observability = get_observability_manager()
            app.state.observability = observability
            await observability.start_monitoring()
            logger.info("Started observability monitoring")

            yield
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
        finally:
            # Shutdown observability monitoring
            if hasattr(app.state, "observability"):
                try:
                    await app.state.observability.stop_monitoring()
                    logger.info("Stopped observability monitoring")
                except Exception as e:
                    logger.error(f"Error stopping observability monitoring: {e}")

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
        response = await call_next(request)
        return response

    # Production monitoring middleware
    @app.middleware("http")
    async def monitoring_middleware(request: Request, call_next):
        """Middleware to collect request metrics and system performance data."""
        import time

        import psutil

        from buttermilk.monitoring import get_metrics_collector

        start_time = time.time()

        # Update system metrics before processing request
        try:
            metrics_collector = get_metrics_collector()
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()

            # Count active WebSocket connections (approximation)
            websocket_connections = 0
            if hasattr(request.app.state, "flow_runner") and hasattr(request.app.state.flow_runner, "session_manager"):
                websocket_connections = len(request.app.state.flow_runner.session_manager.sessions)

            metrics_collector.update_system_metrics(
                memory_mb=memory.used / 1024 / 1024,
                cpu_percent=cpu_percent,
                websocket_connections=websocket_connections
            )
        except Exception as e:
            logger.debug(f"Error updating system metrics: {e}")

        # Process request
        response = await call_next(request)

        # Log request duration for performance monitoring
        duration = time.time() - start_time
        if duration > 1.0:  # Log slow requests
            logger.info(f"Slow request: {request.method} {request.url.path} took {duration:.2f}s")

        return response

    # WebSocket endpoint - essential for frontend terminal functionality
    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for client communication with the WebUIAgent.

        Args:
            websocket: WebSocket connection
            session_id: Unique identifier for this client session

        """
        # Accept the WebSocket connection
        if websocket.client_state == WebSocketState.CONNECTING:
            await websocket.accept()
            logger.info(f"[WEBSOCKET] Connection accepted for session {session_id}")
        else:
            logger.warning(f"[WEBSOCKET] Unexpected WebSocket state {websocket.client_state} for session {session_id}")
        flow_runner: FlowRunner = websocket.app.state.flow_runner
        if not (session := await flow_runner.get_websocket_session_async(session_id=session_id, websocket=websocket)):
            logger.error(f"[WEBSOCKET] Session {session_id} not found.")
            await websocket.close()
            raise HTTPException(status_code=404, detail="Session not found")

        # Start session metrics tracking
        from buttermilk.monitoring import get_metrics_collector
        metrics_collector = get_metrics_collector()
        flow_name = getattr(session, "flow_name", "unknown")
        metrics_collector.start_session_tracking(session_id, flow_name)

        task = None
        # Listen for messages from the client
        token = session_id_var.set(session_id)
        logger.debug(f"[WEBSOCKET] Monitoring UI for session {session_id}")
        async for run_request in session.monitor_ui():
            try:
                logger.info(f"[WEBSOCKET] Received RunRequest in websocket handler: flow={run_request.flow}, session={session_id}")
                await asyncio.sleep(0.1)
                # Track session activity
                metrics_collector.update_session_activity(session_id)

                # This loop internally feeds the groupchat with messages from the client.
                # The only message we receive is a run_request -- which we then
                # use to create a new flow.
                logger.info(f"Creating flow task for '{run_request.flow}' in session {session_id}")
                logger.info(f"[WEBSOCKET] Before creating task - session.websocket: {session.websocket}")
                task = asyncio.create_task(flow_runner.run_flow(
                    run_request=run_request,
                    wait_for_completion=False,
                ))
                logger.info(f"[WEBSOCKET] Task created: {task}")

            except WebSocketDisconnect:
                logger.info(f"Client {session_id} disconnected.")
                break
            except Exception as e:
                # Track error in session metrics
                metrics_collector.update_session_activity(session_id, error_occurred=True)
                msg = f"Error receiving/processing client message for {session_id}: {e}"
                raise FatalError(msg) from e
            finally:
                session_id_var.reset(token)

        # End session tracking
        metrics_collector.end_session_tracking(session_id)

        # Handle client disconnect - try reconnection first, cleanup only if needed
        try:
            if hasattr(flow_runner, "session_manager"):
                # Try to transition to RECONNECTING status instead of immediate cleanup
                reconnect_enabled = await flow_runner.session_manager.handle_client_disconnect(session_id)
                if reconnect_enabled:
                    logger.info(f"Session {session_id} transitioned to RECONNECTING after WebSocket disconnect")
                else:
                    logger.info(f"Session {session_id} cleaned up after WebSocket disconnect (reconnection not applicable)")
        except Exception as e:
            logger.warning(f"Error handling session {session_id} disconnect: {e}")

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

        if not hasattr(flow_runner, "session_manager"):
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

        if not hasattr(flow_runner, "session_manager"):
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

        if not hasattr(flow_runner, "session_manager"):
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

    # Add monitoring router immediately (production infrastructure)
    app.include_router(monitoring_router)
    logger.info("Monitoring router added for production observability")

    # Create lazy middleware for deferred routes
    lazy_manager.create_lazy_middleware()

    # Include flow_data_router directly since its websocket endpoint is now directly on app
    app.include_router(flow_data_router)
    logger.info("Flow data router included directly.")

    logger.info("Heavy routes deferred - will load on first request")

    # Set up static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app
