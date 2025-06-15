"""Main command-line interface (CLI) entry point for the Buttermilk application.

This script serves as the primary interface for launching Buttermilk in various
modes. It uses Hydra for configuration management, allowing settings to be
defined in YAML files and overridden via command-line arguments.

Based on the configuration, this script can:
- Run a Buttermilk flow directly in the console for interactive use (`console` mode).
- Start a FastAPI web server to expose Buttermilk flows via an HTTP API (`api` mode).
- Create batch jobs by adding multiple `RunRequest` instances to a queue (`batch` mode).
- Process jobs from a queue in a worker-like fashion (`batch_run` mode).
- Launch a Streamlit web application for a graphical user interface (`streamlit` mode).
- Start a Google Cloud Pub/Sub listener for message-driven flow execution (`pub/sub` mode,
  potentially delegating to `batch_cli.main`).
- Run a Slack bot that interacts with Buttermilk flows (`slackbot` mode).

It initializes a global `BM` (Buttermilk) instance with the loaded configuration,
which provides access to shared resources like LLM clients, cloud connections,
and secret management. A `FlowRunner` instance is also created to manage the
execution of defined flows.
"""

import asyncio
import os  # For setting environment variables (e.g., Slack tokens)

import hydra  # For configuration management
import uvicorn  # For running the FastAPI server
from omegaconf import DictConfig, OmegaConf  # Hydra's configuration objects

from buttermilk import BM  # The global Buttermilk instance type
from buttermilk._core import (
    # dmrc as DMRC,
    logger,  # Centralized logger
)
from buttermilk._core.types import RunRequest
from buttermilk.agents.ui.console import CLIUserAgent
from buttermilk.api.flow import create_app as create_fastapi_app
from buttermilk.runner.flowrunner import FlowRunner


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(conf: DictConfig) -> None:
    """Main application entry point, configured and launched by Hydra.

    This function initializes the Buttermilk environment (`BM` instance) and a
    `FlowRunner` based on the Hydra configuration (`conf`). It then determines
    the operational mode (e.g., "console", "api", "batch", "slackbot") from the
    configuration and starts the corresponding application logic.

    Args:
        conf (DictConfig): The configuration object loaded and populated by Hydra.
            This OmegaConf `DictConfig` contains nested configurations for various
            parts of the application, such as `bm` (for the global Buttermilk settings),
            `run` (for the `FlowRunner` and operational mode), and mode-specific
            parameters like `flow`, `record_id`, `prompt` for console mode.

    """
    # Resolve OmegaConf to a plain Python dictionary for Pydantic model instantiation
    resolved_cfg_dict = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)

    # Initialize the global Buttermilk instance (bm) with its configuration section
    if "bm" not in resolved_cfg_dict or not isinstance(resolved_cfg_dict["bm"], dict):
        raise ValueError("Hydra configuration must contain a 'bm' dictionary for Buttermilk initialization.")
    bm = BM(**resolved_cfg_dict["bm"])  # type: ignore # Assuming dict matches BM fields
    # Set the singleton BM instance
    from buttermilk._core.dmrc import set_bm

    set_bm(bm)  # Set the Buttermilk instance using the singleton pattern

    # Initialize FlowRunner with its configuration section (e.g., conf.run)
    if "run" not in conf:  # Check on original conf as model_validate expects OmegaConf DictConfig
        raise ValueError("Hydra configuration must contain a 'run' section for FlowRunner.")
    flow_runner: FlowRunner = FlowRunner.model_validate(conf.run)
    flow_runner.flows = conf.flows

    # Branch execution based on the configured UI mode.
    match flow_runner.mode:
        case "console":
            ui = CLIUserAgent()
            # Prepare the RunRequest with command-line parameters
            run_request = RunRequest(
                ui_type=conf.ui,
                flow=conf.get("flow"),
                record_id=conf.get("record_id", ""),
                prompt=conf.get("prompt", ""),
                uri=conf.get("uri", ""),
                callback_to_ui=ui.callback_to_ui,
            )

            # Run the flow synchronously
            logger.info(f"Running flow '{conf.flow}' in console mode...")
            asyncio.run(flow_runner.run_flow(run_request=run_request, wait_for_completion=True))
            logger.info(f"Flow '{run_request.flow}' finished.")

        case "batch":
            logger.info("Creating batch jobs...")
            asyncio.run(flow_runner.create_batch(flow_name=conf.get("flow"), max_records=conf.get("max_records", None)))

        case "batch_run":
            # Run batch jobs from the queue
            # max_jobs controls how many jobs to process before exiting
            # Each job gets a completely fresh orchestrator instance to ensure
            # no state is shared between jobs, preventing cross-contamination
            # This is critical for research integrity where old state might affect results
            max_jobs = conf.get("max_jobs", 5)  # Get max_jobs from config or default to 5
            ui = CLIUserAgent()

            logger.info(f"Running in batch mode with max_jobs={max_jobs}...")
            asyncio.run(flow_runner.run_batch_job(max_jobs=max_jobs, callback_to_ui=ui.make_callback()))

        case "streamlit":
            # Starts the Streamlit web interface.
            logger.info("Starting Streamlit interface...")
            try:
                from buttermilk.web.streamlit_frontend.app import create_dashboard_app
                # create_dashboard_app is expected to configure and run the Streamlit app.
                # It might need access to flow_runner or specific flow configurations.
                streamlit_app_manager = create_dashboard_app(flow_runner=flow_runner)  # Pass FlowRunner
                asyncio.run(streamlit_app_manager.run())  # Assuming create_dashboard_app returns an object with a run method
            except ImportError as e_streamlit:
                logger.error(f"Failed to import Streamlit components: {e_streamlit!s}. Is Streamlit installed?")
            except Exception as e_streamlit_start:
                logger.error(f"Error starting Streamlit interface: {e_streamlit_start!s}", exc_info=True)

        case "api":
            # Starts a FastAPI web server.
            logger.info("Starting FastAPI API server...")
            # The FastAPI app needs access to bm_instance and flow_runner to handle API requests.
            # These are typically passed to the app creation function.
            fastapi_app = create_fastapi_app(
                bm=bm,  # Pass the global BM instance
                flows=flow_runner,  # Pass the FlowRunner
            )

            # Verify app is ready instead of sleeping
            logger.debug("Verifying FastAPI app readiness...")
            if not hasattr(fastapi_app.state, "flow_runner") or not fastapi_app.state.flow_runner:
                raise RuntimeError("FlowRunner not properly initialized in FastAPI app state")
            if not hasattr(fastapi_app.state, "bm") or not fastapi_app.state.bm:
                raise RuntimeError("BM instance not properly initialized in FastAPI app state")
            logger.debug("FastAPI app readiness verified")

            logger.info("Configuring Uvicorn server for FastAPI app...")
            uvicorn_config = uvicorn.Config(
                app=fastapi_app,
                host=str(conf.get("host", "0.0.0.0")),  # Host from config or default
                port=int(conf.get("port", 8000)),    # Port from config or default
                reload=bool(conf.get("reload", False)),  # Hot reloading (dev only)
                log_level=str(conf.get("log_level", "info")).lower(),
                access_log=True,  # Enable access logs
                workers=int(conf.get("workers", 1)),  # Number of worker processes
            )
            api_server = uvicorn.Server(config=uvicorn_config)
            logger.info(f"FastAPI server starting on http://{uvicorn_config.host}:{uvicorn_config.port}")
            try:
                api_server.run()  # This is a blocking call
            except KeyboardInterrupt:
                logger.info("FastAPI server shutting down due to KeyboardInterrupt...")
            finally:
                logger.info("FastAPI server stopped.")

        case "pub/sub":
            # Starts a Google Cloud Pub/Sub listener.
            # This mode might involve running a worker that processes messages from a Pub/Sub topic.
            # The original code delegated to `batch_cli.main`. If `batch_cli.main` is designed
            # to handle Pub/Sub listening when `conf.ui` (or similar) indicates pub/sub mode,
            # then this delegation is appropriate.
            # Ensure `batch_cli.main` is compatible with being called this way.
            logger.info("Pub/Sub mode: Initializing Pub/Sub listener or batch CLI...")
            try:
                from buttermilk.runner.batch_cli import main as batch_cli_main  # Assuming this handles pub/sub logic
                # Pass the already loaded and resolved Hydra config.
                # batch_cli_main might need adaptation if it expects to run @hydra.main itself.
                batch_cli_main(conf)  # This call might be synchronous or start an async loop.
            except ImportError:
                logger.error("Failed to import `buttermilk.runner.batch_cli`. Pub/Sub mode cannot start.")
            except Exception as e_pubsub:
                logger.error(f"Error in Pub/Sub mode execution: {e_pubsub!s}", exc_info=True)

        case "slackbot":
            # Starts a Slack bot integration.
            logger.info("Starting Slackbot mode...")

            # Retrieve Slack tokens securely from bm.credentials
            slack_creds = bm.credentials
            if not isinstance(slack_creds, dict):
                raise TypeError(f"Expected bm.credentials to be a dict, got {type(slack_creds)}")

            slack_bot_token = slack_creds.get("MODBOT_TOKEN")  # Standard bot token
            slack_app_token = slack_creds.get("SLACK_APP_TOKEN")  # Socket Mode app-level token

            if not slack_bot_token or not slack_app_token:
                raise ValueError("Missing MODBOT_TOKEN or SLACK_APP_TOKEN in credentials. Check secrets configuration.")

            # Set environment variables for Slack Bolt library, if it relies on them.
            # Alternatively, pass tokens directly to initialize_slack_bot if supported.
            os.environ["SLACK_BOT_TOKEN"] = slack_bot_token
            os.environ["SLACK_APP_TOKEN"] = slack_app_token

            from buttermilk.runner.slackbot import initialize_slack_bot  # Slack bot initialization utility

            event_loop = asyncio.get_event_loop()
            # Queue for managing asyncio tasks created by Slack event handlers
            orchestrator_tasks = asyncio.Queue()  # type: ignore

            slack_bolt_app, slack_bolt_handler = initialize_slack_bot(
                bot_token=slack_bot_token,
                app_token=slack_app_token,
                loop=event_loop,
            )

            # Start the Slack Bolt handler in a background task
            _ = event_loop.create_task(slack_bolt_handler.start_async())

            async def runloop():
                """Registers handlers and keeps the main loop running for the Slack bot."""
                # Register the specific Buttermilk command/event handlers with the Bolt app.
                # This connects Slack events (like slash commands) to Buttermilk flow execution.
                from buttermilk.runner.slackbot import register_handlers
                await register_handlers(
                    slack_app=slack_bolt_app,
                    flows=flow_runner.flows,
                    orchestrator_tasks=orchestrator_tasks,
                )
                logger.info("Slack handlers registered. Buttermilk Slackbot is ready.")
                # Keep the event loop running; Slack events will drive operations.
                while True:
                    await asyncio.sleep(3600)  # Wake up periodically or rely on events

            try:
                event_loop.run_until_complete(runloop())
            except KeyboardInterrupt:
                logger.info("Slackbot received KeyboardInterrupt. Shutting down...")
            finally:
                # TODO: Implement graceful shutdown for Slackbot (e.g., stop handler, wait for tasks)
                if not event_loop.is_closed():
                    event_loop.close()
                logger.info("Slackbot event loop closed.")

        case _:
            # Handles any unsupported modes specified in the configuration.
            raise ValueError(f"Unsupported run mode in configuration: '{flow_runner.mode}'. Check 'run.mode' in your Hydra config.")


if __name__ == "__main__":
    # This block executes if the script is run directly (e.g., `python -m buttermilk.runner.cli`).
    # Hydra's `@hydra.main` decorator handles parsing command-line arguments
    # and loading the configuration specified by `config_path` and `config_name`.
    main()
