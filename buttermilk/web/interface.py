"""Interface module for integrating web UIs with the Buttermilk application.
Provides functions to create and start FastAPI and Streamlit interfaces.
"""

import asyncio  # Add asyncio import
import sys

import streamlit as st
from hydra import compose, initialize

from buttermilk.runner.flowrunner import FlowRunner


def create_streamlit_frontend(flows=None):
    """Create a Streamlit frontend application for Buttermilk."""
    from buttermilk.web.streamlit_frontend.app import create_dashboard_app

    if flows is None:
        # Load configuration using Hydra via cache (sync call is fine here)
        # Use empty tuple for overrides if none provided, needed for caching key
        hydra_overrides = tuple(sys.argv[1:]) if len(sys.argv) > 1 else tuple()
        flows = get_cached_flow_runner(hydra_overrides)

    return create_dashboard_app(flows)


def _load_hydra_config(overrides: list = []) -> FlowRunner:
    """Load Hydra configuration, similar to how it's done in cli.py"""
    print(f"Loading Hydra config with overrides: {overrides}")  # Debug print
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(config_name="config", overrides=overrides)

    flow_runner = FlowRunner.model_validate(cfg)
    flow_runner.bm.setup_instance()  # Ensure instance is setup
    print("Hydra config loaded and FlowRunner validated.")  # Debug print
    return flow_runner


@st.cache_resource
def get_cached_flow_runner(overrides: tuple) -> FlowRunner:
    """Loads Hydra config and returns a cached FlowRunner instance."""
    # Convert tuple back to list for Hydra
    override_list = list(overrides)
    print(f"Cache miss or first run. Initializing FlowRunner with overrides: {override_list}")
    # The actual config loading happens here, only on cache miss
    return _load_hydra_config(override_list)


# This function now sets up and runs the async Streamlit app
def run_streamlit_async():
    """Sets up and runs the asynchronous Streamlit frontend."""
    # --- SET PAGE CONFIG FIRST ---
    st.set_page_config(
        page_title="Buttermilk Dashboard",
        page_icon="🥛",
        layout="wide",
    )

    # Get Hydra overrides from command line arguments (passed after '--')
    # Convert to tuple for caching
    hydra_overrides = tuple(sys.argv[1:])

    # Get the cached FlowRunner instance (sync call is fine for setup)
    flow_runner = get_cached_flow_runner(hydra_overrides)

    # Create the Streamlit app instance (sync call)
    app = create_streamlit_frontend(flow_runner)

    # Run the app's async run() method using asyncio.run()
    print("Starting asynchronous Streamlit app run...")
    try:
        asyncio.run(app.run())
    except Exception as e:
        print(f"Error running Streamlit app: {e}")  # Log top-level errors
        # Potentially re-raise or handle cleanup
        raise


# Allow this file to be run directly with streamlit
if __name__ == "__main__":
    # Ensure we run the async version when the script is executed
    run_streamlit_async()
