#!/usr/bin/env python
"""Development server script for running the FastAPI + Jinja2 + htmx + Tailwind dashboard.
This is a standalone script that runs the dashboard as a separate application.
"""
import os
import sys
from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import our dashboard app module
from buttermilk.web.fastapi_frontend.app import DashboardApp

# Define paths
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


class MockFlowRunner:
    """A mock FlowRunner class for development/testing purposes"""

    def __init__(self):
        """Initialize with some sample flows"""
        self.flows = {
            "sample_flow": MockFlow("Sample Flow"),
            "judge_flow": MockFlow("Judge Flow"),
            "analytics_flow": MockFlow("Analytics Flow"),
        }
        self.bm = MockBM()

    async def run_flow(self, flow_name=None, run_request=None):
        """Mock method to simulate running a flow"""
        print(f"Mock running flow: {flow_name}")

        # Create a simple callback function that just logs messages
        async def mock_callback(message):
            """Mock callback for handling messages from the UI"""
            print(f"Received message: {message}")
            
            # We could add more sophisticated mock responses here if needed
            # But for now, we'll keep it simple

        return mock_callback


class MockFlow:
    """A mock Flow class for development/testing purposes"""

    def __init__(self, name):
        """Initialize with a name and sample parameters"""
        self.name = name
        self.parameters = {
            "criteria": ["accuracy", "clarity", "relevance"],
        }
        # Create mock data that can be accessed by prepare_step_df
        self.data = {
            "mock_data": {
                "record_ids": ["record1", "record2", "record3", "record4"],
                "index": ["record1", "record2", "record3", "record4"]
            }
        }
        
    async def get_record_ids(self):
        """Mock method to get record IDs"""
        import pandas as pd
        df = pd.DataFrame({
            "content": ["Sample content 1", "Sample content 2", "Sample content 3", "Sample content 4"]
        }, index=["record1", "record2", "record3", "record4"])
        return df


class MockBM:
    """A mock BM class for development/testing purposes"""

    def __init__(self):
        """Initialize with minimal requirements"""

    def run_query(self, sql):
        """Mock method to return sample data"""
        import pandas as pd
        return pd.DataFrame({
            "flow_name": ["sample_flow", "judge_flow"],
            "record_id": ["record1", "record2"],
            "criteria": ["accuracy", "clarity"],
            "agent_name": ["judge", "scorer"],
            "score": [0.85, 0.72],
            "created_at": ["2025-04-29", "2025-04-28"],
        })


def create_app():
    """Create a standalone FastAPI application for development/testing"""
    # Create a mock flow runner
    flows = MockFlowRunner()

    # Create the dashboard app
    dashboard = DashboardApp(flows)
    app = dashboard.get_app()

    # Add static files directly for standalone mode
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


if __name__ == "__main__":
    print("Starting FastAPI development server...")
    print("Dashboard will be available at: http://localhost:8000")

    # Run the server
    uvicorn.run(
        "run:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["buttermilk/web/fastapi_frontend"],
    )
