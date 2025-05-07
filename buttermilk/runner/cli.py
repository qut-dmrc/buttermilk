"""Main entry point for the Buttermilk application.

This script handles command-line arguments, loads configuration using Hydra,
initializes the application components (including the Buttermilk instance 'bm'),
and starts the appropriate user interface (UI) mode based on the configuration.

Supported UI modes:
- console: Runs a specified flow directly in the console.
- api: Starts a FastAPI web server to interact with flows via HTTP requests.
- pub/sub: Starts a Google Cloud Pub/Sub listener (details TBD).
- slackbot: Starts a Slack bot to interact with flows via Slack commands/events.
"""

import asyncio
import os

import hydra
import uvicorn
from omegaconf import DictConfig  # Import DictConfig for type hinting

from buttermilk._core.types import RunRequest
from buttermilk.api.flow import create_app
from buttermilk.api.job_queue import JobQueueClient
from buttermilk.bm import BM, logger
from buttermilk.runner.flowrunner import FlowRunner
from buttermilk.runner.slackbot import register_handlers


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main application function orchestrated by Hydra.

    Args:
        cfg: The configuration object loaded and populated by Hydra.
             Expected to conform to a structure allowing instantiation of
             'bm' (Buttermilk instance), 'flows', 'ui', etc.
             Note: While typed as DictConfig, Hydra instantiates objects within it.
             Using a more specific OmegaConf schema or a dataclass (like OrchestratorProtocol
             attempted before) could improve type safety if cfg structure is stable.

    """
    # Hydra automatically instantiates objects defined in the configuration files (e.g., bm, flows).
    # and any overrides (like `conf/flows/batch.yaml` when running `python -m buttermilk.runner.cli flows=batch`).

    flow_runner: FlowRunner = FlowRunner.model_validate(cfg)  # hydra.utils.instantiate(cfg)
    bm: BM = flow_runner.bm  # Access the instantiated Buttermilk core instance.

    # Perform essential setup for the Buttermilk instance *after* it's been instantiated by Hydra.
    # This might involve loading credentials, setting up logging, initializing API clients, etc.
    bm.setup_instance()

    logger.info(f"Starting UI: {flow_runner.ui}...")
    # Branch execution based on the configured UI mode.
    match flow_runner.ui:
        case "console":
            # Run a flow directly in the console
            flow_name = cfg.flow

            # Prepare the RunRequest with command-line parameters
            run_request = RunRequest(flow=cfg.get("flow"),
                record_id=cfg.get("record_id", ""),
                prompt=cfg.get("prompt", ""),
                uri=cfg.get("uri", ""),
            )

            # Run the flow synchronously
            bm.logger.info(f"Running flow '{cfg.flow}' in console mode...")
            asyncio.run(flow_runner.run_flow(run_request=run_request, wait_for_completion=True))
            bm.logger.info(f"Flow '{cfg.flow}' finished.")

        case "batch":
            # Run a batch job using the batch runner
            from buttermilk.runner.batch_cli import main as batch_main

            bm.logger.info("Running in batch mode...")
            # We're already in the hydra context, so we can just call the main function
            batch_main(cfg)

        case "streamlit":
            # Start the Streamlit interface

            bm.logger.info("Starting Streamlit interface...")
            try:
                # This function provides guidance on how to run the app with streamlit CLI
                from buttermilk.web.streamlit_frontend.app import create_dashboard_app
                app = create_dashboard_app(flows=flow_runner)
                # Run the app
                asyncio.run(app.run())
            except Exception as e:
                bm.logger.error(f"Error starting Streamlit interface: {e}")

        case "api":

            # Inject the instantiated flows and orchestrator classes into the FastAPI app's state
            # so that API endpoints can access them.
            # TODO: Using app.state is simple but can be considered a form of global state.
            #       Dependency injection frameworks (like FastAPI's Depends) might offer cleaner alternatives
            #       for larger applications.

            # Create the FastAPI app with dependencies
            bm.logger.info("Attempting to create FastAPI app...")
            app = create_app(
                bm=bm,
                flows=flow_runner,
            )

            # --- WORKAROUND for potential bm async init timing ---
            # If bm.setup_instance starts background tasks, give them time.
            # This is fragile; a better fix involves awaiting readiness.
            import time
            time.sleep(2)
            bm.logger.info("Configuring API server...")
            # Configure Uvicorn server
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=8000,
                reload=True,  # Set to True if you want hot reloading
                log_level="info",
                access_log=True,
                workers=1,
            )

            bm.logger.info("Creating server instance...")
            # Create and run the server
            server = uvicorn.Server(config)
            bm.logger.info("Starting API server...")

            try:
                server.run()
            except KeyboardInterrupt:
                bm.logger.info("Shutting down API server...")

        case "pub/sub":
            # Start a listener for Google Cloud Pub/Sub messages.

            worker = JobQueueClient(
                flow_runner=flow_runner,
                max_concurrent_jobs=1,
            )
            logger.info("Started job worker in FastAPI application")
            asyncio.run(worker.fetch_and_run_task())

            bm.logger.info("Pub/Sub listener stopped.")

        case "slackbot":
            # Start a Slack bot integration.
            bm.logger.info("Starting Slackbot...")

            # Securely retrieve Slack tokens from the Buttermilk credentials store.
            creds = bm.credentials
            if not isinstance(creds, dict):
                # Ensure credentials are in the expected format.
                raise TypeError(f"Expected credentials to be a dict, got {type(creds)}")

            bot_token = creds.get("MODBOT_TOKEN")
            app_token = creds.get("SLACK_APP_TOKEN")  # Socket Mode token

            if not bot_token or not app_token:
                raise ValueError("Missing MODBOT_TOKEN or SLACK_APP_TOKEN in credentials. Check config/secrets.")

            # Set environment variables required by the Slack Bolt library.
            # TODO: Consider passing tokens directly instead of using environment variables if preferred.
            os.environ["SLACK_BOT_TOKEN"] = bot_token
            os.environ["SLACK_APP_TOKEN"] = app_token

            from buttermilk.runner.slackbot import initialize_slack_bot

            loop = asyncio.get_event_loop()

            # Queue for managing background tasks initiated by Slack events.
            orchestrator_tasks: asyncio.Queue[asyncio.Task] = asyncio.Queue()

            # Initialize the Slack Bolt app and its handler.
            slack_app, handler = initialize_slack_bot(bot_token=bot_token, app_token=app_token, loop=loop)  # Pass tokens and loop

            # Start the Slack Bolt handler in the background.
            _ = loop.create_task(handler.start_async())  # Use underscore if task result isn't needed immediately.

            async def runloop():
                """Registers handlers and keeps the main loop running for the Slack bot."""
                # Register the specific Buttermilk command/event handlers with the Bolt app.
                # This connects Slack events (like slash commands) to Buttermilk flow execution.
                await register_handlers(
                    slack_app=slack_app,
                    flows=flow_runner.flows,
                    orchestrator_tasks=orchestrator_tasks,
                )
                bm.logger.info("Slack handlers registered. Bot is ready.")
                # Keep the event loop running indefinitely for the bot.
                # TODO: Implement a graceful shutdown mechanism (e.g., catching SIGINT/SIGTERM).
                while True:
                    await asyncio.sleep(3600)  # Sleep for a long time; loop is driven by Slack events.

            # Run the main application loop until completion (which is indefinite for the bot).
            try:
                loop.run_until_complete(runloop())
            except KeyboardInterrupt:
                bm.logger.info("Slackbot shutting down...")
            finally:
                # TODO: Add cleanup logic here if needed (e.g., close connections, wait for tasks).
                loop.close()

        case _:
            # Handle unexpected UI modes.
            raise ValueError(f"Unsupported UI mode specified in configuration: {flow_runner.ui}")


if __name__ == "__main__":
    # This block executes when the script is run directly.
    # Hydra takes over argument parsing and configuration loading here.
    main()
