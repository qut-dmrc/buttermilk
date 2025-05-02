#!/bin/bash
# Run Streamlit dashboard for Buttermilk

# Ensure we're running from the correct directory
cd "$(dirname "$0")/../../.." || exit

# Run the Streamlit app
uv run streamlit run buttermilk/web/streamlit_frontend/run.py
