"""Interface module for integrating web UIs with the Buttermilk application.
Provides functions to create and start FastAPI and Streamlit interfaces.
"""

import importlib
import sys

import uvicorn

from buttermilk.runner.flowrunner import FlowRunner


def create_fastapi_frontend(flows: FlowRunner):
    """Create a FastAPI frontend application for Buttermilk.
    
    Args:
        flows: FlowRunner instance or None (will create a new instance if None)
        
    Returns:
        FastAPI application instance

    """
    from buttermilk.web.fastapi_frontend.app import create_dashboard_app

    return create_dashboard_app(flows)


def create_streamlit_frontend(flows: FlowRunner):
    """Create a Streamlit frontend application for Buttermilk.
    
    Args:
        flows: FlowRunner instance or None (will create a new instance if None)
        
    Returns:
        StreamlitDashboardApp instance

    """
    from buttermilk.web.streamlit_frontend.app import create_dashboard_app

    return create_dashboard_app(flows)


def run_fastapi(host="0.0.0.0", port=8000, reload=False):
    """Run the FastAPI frontend with uvicorn.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Whether to enable auto-reload

    """
    # Setup the Buttermilk configuration and flow runner
    flows = FlowRunner(config)

    # Create the app
    app = create_fastapi_frontend(flows)

    # Run with uvicorn
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )


def run_streamlit():
    """Run the Streamlit frontend.
    
    This function does not directly run Streamlit, but instead ensures
    that the Streamlit app can access the proper Buttermilk configuration
    and flows. You should run this through the streamlit CLI:
    
    streamlit run buttermilk/web/interface.py
    """
    # Check if we're being run via streamlit
    if importlib.util.find_spec("streamlit.web.bootstrap") is None:
        print("This script should be run using the streamlit CLI:")
        print("streamlit run buttermilk/web/interface.py")
        sys.exit(1)

    # Setup the Buttermilk configuration and flow runner
    flows = FlowRunner(config)

    # Create and run the Streamlit app
    app = create_streamlit_frontend(flows)
    app.run()


# Allow this file to be run directly with streamlit
if __name__ == "__main__":
    run_streamlit()
