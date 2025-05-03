from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from buttermilk.api.job_queue import JobQueueClient
from buttermilk.bm import logger
from buttermilk.runner.job_worker import start_job_worker
from buttermilk.web.activity_tracker import get_instance as get_activity_tracker
from buttermilk.web.fastapi_frontend.routes import DashboardRoutes


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


# Define the base directory for the FastAPI app
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


def create_dashboard_app(flows) -> FastAPI:
    """Create and configure the dashboard application
    
    Args:
        flows: FlowRunner or mock runner instance with workflow configurations
        
    Returns:
        FastAPI: The FastAPI application

    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan event handler for startup and shutdown events.
        """
        try:
            # Startup event
            job_queue = JobQueueClient()
            worker = await start_job_worker(
                flow_runner=flows,
                job_queue=job_queue,
                max_concurrent_jobs=1,
            )
            app.state.job_worker = worker
            logger.info("Started job worker in FastAPI application")
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

    # Set up templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Set up static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Initialize routes
    dashboard_routes = DashboardRoutes(templates, flows)

    # Include the API routes
    app.include_router(dashboard_routes.get_router())

    return app
