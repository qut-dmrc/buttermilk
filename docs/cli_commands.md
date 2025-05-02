# Buttermilk CLI Commands

This document outlines how to run Buttermilk with different UI modes using the command-line interface.

## Running with FastAPI Frontend

```bash
python -m buttermilk.runner.cli ui=api
```

## Running with Streamlit Frontend

```bash
# Method 1: Use the Streamlit UI mode in CLI
python -m buttermilk.runner.cli ui=streamlit

# Method 2: Run the interface directly with Streamlit
streamlit run buttermilk/web/interface.py
```

## Running a Flow in Console Mode

```bash
python -m buttermilk.runner.cli ui=console flow=<flow_name> record_id=<record_id>
```

## Running with Pub/Sub

```bash
python -m buttermilk.runner.cli ui=pub/sub
```

## Running with Slackbot

```bash
python -m buttermilk.runner.cli ui=slackbot
