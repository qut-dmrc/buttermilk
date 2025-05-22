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
import os # For setting environment variables (e.g., Slack tokens)
from uuid import uuid4 # For generating unique IDs

import hydra # For configuration management
import uvicorn # For running the FastAPI server
from omegaconf import DictConfig, OmegaConf # Hydra's configuration objects

from buttermilk import BM # The global Buttermilk instance type
from buttermilk._core import (
    # dmrc as DMRC,  # noqa - DMRC module might be for specific singleton patterns
    logger, # Centralized logger
)
from buttermilk._core.config import FatalError # Custom exception
# from buttermilk._core.contract import OmegaConf # OmegaConf is imported from omegaconf directly
from buttermilk._core.types import RunRequest # Pydantic model for run requests
from buttermilk.agents.ui.console import CLIUserAgent # Console UI agent
from buttermilk.api.flow import create_app as create_fastapi_app # FastAPI app factory
from buttermilk.api.job_queue import JobQueueClient # Client for interacting with job queue
from buttermilk.runner.flowrunner import FlowRunner # Manages flow execution
from buttermilk.runner.slackbot import register_handlers as register_slack_handlers # Slack handler registration

# Global Buttermilk instance (bm) will be initialized in main() after Hydra loads config.


async def create_batch(flow_runner: FlowRunner, flow: str, max_records: int | None = None) -> None:
    """Creates a new batch of jobs and publishes them to a queue.

    This function generates a list of `RunRequest` objects based on the specified
    `flow` and `max_records` using the `flow_runner.create_batch` method.
    Each generated `RunRequest` is then published as a job to a queue via
    the `JobQueueClient`.

    Args:
        flow_runner: The `FlowRunner` instance used to generate batch requests.
        flow: The name of the flow for which to create batch jobs.
        max_records: Optional. The maximum number of records or jobs to create
            in this batch. Passed to `flow_runner.create_batch`.

    Raises:
        FatalError: If publishing any job to the queue fails.
    """
    try:
        logger.info(f"Creating batch for flow '{flow}' with max_records: {max_records or 'all'}.")
        # Create the batch of run requests
        batch_requests = await flow_runner.create_batch(flow, max_records)
        if not batch_requests:
            logger.warning(f"No batch requests generated for flow '{flow}'. Nothing to publish.")
            return

        logger.info(f"Generated {len(batch_requests)} job(s) for flow '{flow}'. Enqueuing...")
        job_queue_client = JobQueueClient()

        for request_item in batch_requests:
            job_queue_client.publish_job(request_item)
        logger.info(f"Successfully published {len(batch_requests)} job(s) to the queue for flow '{flow}'.")

    except Exception as e:
        error_msg = f"Failed to create or publish batch job for flow '{flow}': {e!s}"
        logger.error(error_msg, exc_info=True)
        raise FatalError(error_msg) from e


async def run_batch_job(flow_runner: FlowRunner, max_jobs: int = 1) -> None:
    """Pulls and runs a specified number of jobs from the queue.

    This function acts as a worker that processes jobs from a queue one by one.
    For each job, it ensures a fresh state by creating a new `CLIUserAgent`
    instance and then uses the provided `flow_runner` to execute the flow
    defined in the `RunRequest`. Execution for each job waits for completion
    before fetching the next job.

    Args:
        flow_runner: The `FlowRunner` instance to use for executing the flows.
        max_jobs: The maximum number of jobs to pull from the queue and process
            in this invocation. Defaults to 1.

    Raises:
        FatalError: If `max_jobs` is positive but no run requests are found in
            the queue after the first pull attempt.
        Exception: Re-raises exceptions that occur during `flow_runner.run_flow`
                   if not caught and logged by `run_flow` itself.
    """
    try:
        # Initialize a JobQueueClient to pull jobs.
        # max_concurrent_jobs=1 ensures jobs are processed sequentially by this worker.
        job_queue_worker = JobQueueClient(max_concurrent_jobs=1)
        jobs_processed_count = 0

        logger.info(f"Starting batch job run. Will process up to {max_jobs} job(s).")

        while jobs_processed_count < max_jobs:
            run_request = await job_queue_worker.pull_single_task()

            if not run_request: # No job pulled from the queue
                if jobs_processed_count == 0 and max_jobs > 0:
                    # If this was the first attempt and no job was found, it's a fatal issue for this worker run.
                    raise FatalError("No run requests found in the queue for batch job processing.")
                logger.info("No more jobs in the queue, or pull limit reached for this worker.")
                break  # Exit loop if no more jobs or if an issue occurred

            # Each job gets a fresh UI agent instance for isolated state if UI interaction is needed.
            # The session_id for the UI agent is different from run_request.session_id (which is for the flow run).
            console_ui = CLIUserAgent(session_id=uuid4().hex)
            run_request.callback_to_ui = console_ui.callback_to_ui # Assign callback for this job

            logger.info(
                f"Processing batch job {jobs_processed_count + 1}/{max_jobs}: "
                f"Flow='{run_request.flow}', JobID='{run_request.job_id}', SessionID='{run_request.session_id}'"
            )
            try:
                # Execute the flow and wait for its completion.
                await flow_runner.run_flow(run_request=run_request, wait_for_completion=True)
                logger.info(f"Successfully completed job '{run_request.job_id}'.")
            except Exception as job_exec_error:
                # Log error for this specific job but allow batch processing to continue with next job.
                logger.error(f"Error running job '{run_request.job_id}': {job_exec_error!s}", exc_info=True)
                # Depending on requirements, could mark job as failed in a database or re-queue with backoff.

            jobs_processed_count += 1

        logger.info(f"Batch job processing run complete. Processed {jobs_processed_count} job(s).")

    except FatalError: # Re-raise FatalErrors from this function or JobQueueClient
        raise
    except Exception as e: # Catch other unexpected errors during batch setup or loop
        logger.error(f"Fatal error during batch job processing run: {e!s}", exc_info=True)
        raise # Re-raise to indicate failure of the batch run itself


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
    bm_instance = BM(**resolved_cfg_dict["bm"]) # type: ignore # Assuming dict matches BM fields

    # Initialize FlowRunner with its configuration section (e.g., conf.run)
    if "run" not in conf: # Check on original conf as model_validate expects OmegaConf DictConfig
        raise ValueError("Hydra configuration must contain a 'run' section for FlowRunner.")
    flow_runner: FlowRunner = FlowRunner.model_validate(conf.run)

    # Set the initialized BM instance as the global singleton
    # This allows other modules to access `bm` via `buttermilk._core.dmrc.get_bm()`.
    from buttermilk._core.dmrc import set_bm 
    set_bm(bm_instance)

    # --- Branch execution based on the configured UI/operational mode ---
    logger.info(f"Buttermilk CLI started in mode: '{flow_runner.mode}'.")
    match flow_runner.mode:
        case "console":
            # Runs a specified flow directly in the console.
            console_ui = CLIUserAgent(session_id=uuid4().hex) # Fresh UI agent for this run
            # Prepare RunRequest from Hydra config (conf.<param> or conf.get("<param>"))
            run_request = RunRequest(
                ui_type=str(conf.ui), # 'ui' key from root or run config
                flow=str(conf.get("flow")), # Expects flow name, e.g., from command line `flow=my_flow`
                record_id=str(conf.get("record_id", "")),
                prompt=str(conf.get("prompt", "")),
                uri=str(conf.get("uri", "")),
                session_id=console_ui.session_id, # Use UI agent's session for this interaction
                callback_to_ui=console_ui.callback_to_ui,
                # Parameters for the flow can be passed via conf.parameters or similar
                parameters=OmegaConf.to_container(conf.get("parameters", {}), resolve=True) # type: ignore
            )
            logger.info(f"Running flow '{run_request.flow}' in console mode...")
            asyncio.run(flow_runner.run_flow(run_request=run_request, wait_for_completion=True))
            logger.info(f"Flow '{run_request.flow}' finished.")

        case "batch":
            # Creates batch jobs and publishes them to a queue.
            logger.info("Batch creation mode: Creating and enqueuing batch jobs...")
            asyncio.run(create_batch(
                flow_runner=flow_runner, 
                flow=str(conf.get("flow")), # Flow name for which to create batch
                max_records=conf.get("max_records") # Optional limit on records for batch
            ))
            logger.info("Batch job creation and enqueuing finished.")

        case "batch_run":
            # Pulls and runs jobs from the queue.
            max_jobs_to_run = conf.get("max_jobs", 1) # How many jobs this worker instance will process
            logger.info(f"Batch run mode: Processing up to {max_jobs_to_run} job(s) from the queue...")
            asyncio.run(run_batch_job(flow_runner=flow_runner, max_jobs=max_jobs_to_run))
            logger.info("Batch run processing finished for this worker instance.")

        case "streamlit":
            # Starts the Streamlit web interface.
            logger.info("Starting Streamlit interface...")
            try:
                from buttermilk.web.streamlit_frontend.app import create_dashboard_app
                # create_dashboard_app is expected to configure and run the Streamlit app.
                # It might need access to flow_runner or specific flow configurations.
                streamlit_app_manager = create_dashboard_app(flow_runner=flow_runner) # Pass FlowRunner
                asyncio.run(streamlit_app_manager.run()) # Assuming create_dashboard_app returns an object with a run method
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
                bm=bm_instance, # Pass the global BM instance
                flows_runner=flow_runner, # Pass the FlowRunner
            )

            # Short delay, as a workaround for potential async initialization timing issues
            # TODO: Replace with a more robust readiness check if needed.
            import time
            time.sleep(1) # Reduced from 2s, check if still needed
            
            logger.info("Configuring Uvicorn server for FastAPI app...")
            uvicorn_config = uvicorn.Config(
                app=fastapi_app,
                host=str(conf.get("host", "0.0.0.0")), # Host from config or default
                port=int(conf.get("port", 8000)),    # Port from config or default
                reload=bool(conf.get("reload", False)), # Hot reloading (dev only)
                log_level=str(conf.get("log_level", "info")).lower(),
                access_log=True, # Enable access logs
                workers=int(conf.get("workers", 1)), # Number of worker processes
            )
            api_server = uvicorn.Server(config=uvicorn_config)
            logger.info(f"FastAPI server starting on http://{uvicorn_config.host}:{uvicorn_config.port}")
            try:
                api_server.run() # This is a blocking call
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
                from buttermilk.runner.batch_cli import main as batch_cli_main # Assuming this handles pub/sub logic
                # Pass the already loaded and resolved Hydra config.
                # batch_cli_main might need adaptation if it expects to run @hydra.main itself.
                batch_cli_main(conf) # This call might be synchronous or start an async loop.
            except ImportError:
                logger.error("Failed to import `buttermilk.runner.batch_cli`. Pub/Sub mode cannot start.")
            except Exception as e_pubsub:
                logger.error(f"Error in Pub/Sub mode execution: {e_pubsub!s}", exc_info=True)

        case "slackbot":
            # Starts a Slack bot integration.
            logger.info("Starting Slackbot mode...")
            
            # Retrieve Slack tokens securely from bm.credentials
            slack_creds = bm_instance.credentials
            if not isinstance(slack_creds, dict):
                raise TypeError(f"Expected bm.credentials to be a dict, got {type(slack_creds)}")
            
            slack_bot_token = slack_creds.get("MODBOT_TOKEN") # Standard bot token
            slack_app_token = slack_creds.get("SLACK_APP_TOKEN")  # Socket Mode app-level token

            if not slack_bot_token or not slack_app_token:
                raise ValueError("Missing MODBOT_TOKEN or SLACK_APP_TOKEN in credentials. Check secrets configuration.")

            # Set environment variables for Slack Bolt library, if it relies on them.
            # Alternatively, pass tokens directly to initialize_slack_bot if supported.
            os.environ["SLACK_BOT_TOKEN"] = slack_bot_token
            os.environ["SLACK_APP_TOKEN"] = slack_app_token

            from buttermilk.runner.slackbot import initialize_slack_bot # Slack bot initialization utility

            event_loop = asyncio.get_event_loop()
            # Queue for managing asyncio tasks created by Slack event handlers
            slack_orchestrator_tasks: asyncio.Queue[asyncio.Task[Any]] = asyncio.Queue() # type: ignore

            slack_bolt_app, slack_bolt_handler = initialize_slack_bot(
                bot_token=slack_bot_token, 
                app_token=slack_app_token, 
                loop=event_loop
            )
            
            # Start the Slack Bolt handler in a background task
            _ = event_loop.create_task(slack_bolt_handler.start_async())

            async def slack_main_loop():
                """Registers Slack event handlers and keeps the main loop running for the bot."""
                # Register Buttermilk-specific command/event handlers with the Bolt app
                await register_slack_handlers(
                    slack_app=slack_bolt_app,
                    flows_config=flow_runner.flows, # Pass flow configurations
                    flow_runner_instance=flow_runner, # Pass FlowRunner instance
                    orchestrator_tasks_queue=slack_orchestrator_tasks, # Pass the task queue
                )
                logger.info("Slack handlers registered. Buttermilk Slackbot is ready.")
                # Keep the event loop running; Slack events will drive operations.
                while True:
                    await asyncio.sleep(3600) # Wake up periodically or rely on events

            try:
                event_loop.run_until_complete(slack_main_loop())
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
