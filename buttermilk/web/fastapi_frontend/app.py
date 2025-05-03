from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from buttermilk.web.fastapi_frontend.routes import DashboardRoutes

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
    app = FastAPI()

    # Set up templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Set up static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Initialize routes
    dashboard_routes = DashboardRoutes(templates, flows)

    # Include the API routes
    app.include_router(dashboard_routes.get_router())

    return app
