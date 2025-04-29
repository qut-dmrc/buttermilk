# Buttermilk Streamlit Frontend

A Streamlit-based web interface for the Buttermilk dashboard.

## Overview

This directory contains a Streamlit implementation of the Buttermilk dashboard, providing the same functionality as the FastAPI frontend but using Streamlit for the UI. Streamlit offers a more interactive and simpler development experience for data applications.

## Features

- Select flows, criteria, and record IDs
- Start flow executions
- View agent chat messages in real-time
- Track workflow progress
- Load and view run history
- Respond to confirmation requests

## Running the Streamlit Dashboard

You can run the Streamlit dashboard using the included shell script:

```bash
./buttermilk/web/streamlit_frontend/run_streamlit.sh
```

Or directly with streamlit:

```bash
streamlit run buttermilk/web/streamlit_frontend/run.py
```

The application will start and automatically open in your default web browser.

## Development

The implementation mirrors the structure of the FastAPI frontend but leverages Streamlit's reactive programming model:

- `app.py` - Contains the `StreamlitDashboardApp` class that implements the dashboard interface
- `run.py` - Contains a standalone mock implementation for development and testing
- `run_streamlit.sh` - Shell script to easily run the application

## Mock Data

When using the `run.py` script directly, it uses mock data to simulate flows, records, and database queries. This allows for testing and development without requiring a full Buttermilk backend setup.
