# Buttermilk CLI Commands

This document outlines the primary ways to launch and interact with Buttermilk using the command-line interface (CLI). The base command for most operations is `python -m buttermilk.runner.cli`.

## Running with FastAPI Frontend (API Mode)

This mode starts the Buttermilk backend with a FastAPI web server, allowing you to interact with Buttermilk programmatically via its API endpoints.

```bash
python -m buttermilk.runner.cli ui=api
```
*   The API server typically runs on `http://127.0.0.1:8000`.
*   API documentation (Swagger UI) is usually available at `http://127.0.0.1:8000/docs`.
*   Alternative documentation (ReDoc) might be at `http://127.0.0.1:8000/redoc`.

This mode is essential for programmatic access, integrations, and running batch processes (see `batch_processing.md`).

## Running with Streamlit Frontend (Web Interface)

This mode launches a Streamlit web application, providing a graphical user interface for interacting with Buttermilk.

```bash
# Method 1: Use the Streamlit UI mode in CLI
python -m buttermilk.runner.cli ui=streamlit

# Method 2: Run the interface directly with Streamlit (if you know the path to the Streamlit app script)
# streamlit run buttermilk/web/interface.py
```
*   Streamlit usually runs on a port like `http://127.0.0.1:8501`. Check your console output for the exact URL.
*   This mode is ideal for users who prefer a visual interface for running flows, managing data, or viewing results.

## Running a Flow in Console Mode

This mode allows you to execute a specific flow for a single record directly in your console. It's useful for testing, debugging, or simple, one-off processing tasks.

```bash
python -m buttermilk.runner.cli ui=console flow=<flow_name> record_id=<record_id> # <any_hydra_overrides>
```
*   Replace `<flow_name>` with the name of the flow you want to run (e.g., `sentiment_analysis_flow`).
*   Replace `<record_id>` with the unique identifier of the record to process (e.g., `doc_001`).
*   You can pass additional Hydra overrides to customize flow parameters or other configurations for this specific run. For example: `flow_parameters.model_name=gpt-4`.
*   Output, logs, and results will typically be printed to the console or saved to locations defined in the flow's configuration.

Example:
```bash
python -m buttermilk.runner.cli ui=console flow=twitter_processing record_id=tweet_12345 flow_parameters.api_key=YOUR_API_KEY
```

## Running in Batch Processing Mode

For processing multiple records with a flow, Buttermilk offers a dedicated batch processing mode via the CLI.

```bash
python -m buttermilk.runner.cli ui=batch flow=<flow_name> # <batch_specific_hydra_overrides>
```
*   This command has several sub-commands and options for creating, listing, and managing batch jobs.
*   For detailed usage, please refer to **[batch_processing.md](batch_processing.md)**.

## Running as a Pub/Sub Worker

This mode starts Buttermilk as a worker process that listens to a Publish/Subscribe message queue (e.g., Google Pub/Sub, RabbitMQ, Kafka) for jobs to execute.

```bash
python -m buttermilk.runner.cli ui=pub/sub
```
*   This is an advanced mode, typically used for distributed, asynchronous task processing in a larger system.
*   Requires specific configuration in your `conf/` directory (e.g., `conf/pubsub/default.yaml`) to connect to the message broker and define worker behavior.

## Running as a Slackbot

This mode allows Buttermilk to operate as a Slackbot, enabling users to trigger flows or query information via Slack commands.

```bash
python -m buttermilk.runner.cli ui=slackbot
```
*   This is an advanced mode for integrating Buttermilk into a Slack workspace.
*   Requires specific configuration in your `conf/` directory (e.g., `conf/slack/default.yaml`) including Slack API tokens and bot settings.

---

For more details on specific configurations or advanced use cases, consult the relevant configuration files in the `conf/` directory and other documentation files. Understanding core concepts in **[concepts.md](concepts.md)** will also be beneficial.
