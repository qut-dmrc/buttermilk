"""Main command-line interface (CLI) entry point for the Buttermilk application.

This script serves as the primary interface for launching Buttermilk in various
modes. It uses Hydra for configuration management, allowing settings to be
defined in YAML files and overridden via command-line arguments.

Based on the configuration, this script can:
- Run a Buttermilk flow directly in the console for interactive use (`console` mode).
- Start a FastAPI web server to expose Buttermilk flows via an HTTP API (`api` mode).
- Create batch jobs by adding multiple `RunRequest` instances to a queue (`batch` mode).
- Process jobs from a queue in a worker-like fashion (`batch_run` mode).
- Combined batch operations: enqueue and/or process jobs (`batch_all` mode).
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
from omegaconf import DictConfig, OmegaConf  # Hydra's configuration objects

from buttermilk import (
    init_async,
    # dmrc as DMRC,
    logger,  # Centralized logger
)
from buttermilk._core.types import RunRequest
from buttermilk.agents.ui.console import CLIUserAgent
from buttermilk.runner.flowrunner import FlowRunner


def _validate_flow_config(conf: DictConfig, mode: str) -> None:
    """Validate that required flow configuration exists for the given mode.

    Args:
        conf: The configuration object
        mode: The operational mode (can be string or RunMode enum)

    Raises:
        ValueError: If required configuration is missing or invalid
    """
    # Convert mode to string if it's an enum
    mode_str = (
        str(mode).split(".")[-1].lower()
        if hasattr(mode, "value")
        else str(mode).lower()
    )

    # Modes that require a flow to be specified
    flow_required_modes = {"console", "batch", "batch_all"}

    if mode_str in flow_required_modes:
        if not hasattr(conf.run, "flow") or not conf.run.flow:
            available_modes = ", ".join(flow_required_modes)
            raise ValueError(
                f"Mode '{mode_str}' requires 'run.flow' to be specified.\n"
                f"Usage: python -m buttermilk.runner.cli run.mode={mode_str} run.flow=<flow_name>\n"
                f"Example: python -m buttermilk.runner.cli run.mode={mode_str} run.flow=trans\n"
                f"Modes requiring flow: {available_modes}"
            )

        # Check if the flow exists in the configuration
        flow_name = conf.run.flow
        if not hasattr(conf.run, "flows") or flow_name not in conf.run.flows:
            available_flows = (
                list(conf.run.flows.keys()) if hasattr(conf.run, "flows") else []
            )
            flows_list = (
                ", ".join(available_flows) if available_flows else "none configured"
            )
            raise ValueError(
                f"Flow '{flow_name}' not found in configuration.\n"
                f"Available flows: {flows_list}\n"
                f"Check your flow configurations in buttermilk/conf/flows/"
            )


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(conf: DictConfig) -> None:  # noqa: PLR0912
    """Main application entry point with async initialization.

    This function initializes the Buttermilk environment using async initialization
    as the primary pathway. It creates the infrastructure and session-scoped BM
    instances, then determines the operational mode and starts the corresponding
    application logic.

    Args:
        conf (DictConfig): The configuration object loaded and populated by Hydra.
            This OmegaConf `DictConfig` contains nested configurations for various
            parts of the application.

    Usage Examples:
        # Run a flow in console mode with default config
        python -m buttermilk.runner.cli run.mode=console run.flow=trans

        # Run batch processing with limit
        python -m buttermilk.runner.cli run.mode=batch run.flow=trans run.limit=100

        # Start API server
        python -m buttermilk.runner.cli run.mode=api

        # Run pipeline mode
        python -m buttermilk.runner.cli run.mode=pipeline

    Hydra Configuration Overrides:
        You can override any configuration parameter using Hydra's dot notation:
        - run.mode=<mode>        : Set operational mode (console, batch, batch_run, batch_all, api, pipeline, streamlit, slackbot)
        - run.flow=<flow_name>   : Specify which flow to run (required for most modes)
        - run.limit=<number>     : Limit number of records/jobs to process
        - run.record_id=<id>     : Run on a specific record (console mode)
        - llms=<config>          : Override LLM configuration
        - storage=<config>       : Override storage configuration

    Available Modes:
        - console    : Run a single flow interactively in the terminal
        - batch      : Create batch jobs and add them to a queue
        - batch_run  : Process jobs from the queue (worker mode)
        - batch_all  : Create and process batch jobs in one command
        - api        : Start FastAPI server for HTTP API access
        - pipeline   : Run data processing pipeline
        - streamlit  : Launch Streamlit web interface
        - slackbot   : Start Slack bot integration

    """
    OmegaConf.resolve(conf)

    # Run async initialization
    bm = asyncio.run(
        init_async(
            config=conf  # Pass the existing Hydra configuration
        )
    )
    conf = bm.cfg  # Use the typed config from BM
    logger.info("Unified bootstrap complete - BM and config ready")

    # Get the mode from config to determine if we need FlowRunner
    mode = conf.run.mode
    logger.info(f"Running in '{mode}' mode")

    # Validate configuration before proceeding
    try:
        _validate_flow_config(conf, mode)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    flow_runner = FlowRunner(flows=conf.run.flows)

    # Set the session-scoped BM for this FlowRunner
    flow_runner.set_session_bm(bm)

    # Branch execution based on the configured UI mode.
    match mode:
        case "console":
            # Console mode: Run a single flow interactively
            logger.info("Console mode: Running single flow interactively")
            ui = CLIUserAgent()

            # Prepare the RunRequest with command-line parameters
            parameters = {}
            if conf.run.record_id:
                parameters["record_id"] = conf.run.record_id
                logger.info(f"Running on specific record_id: {conf.run.record_id}")

            run_request = RunRequest(
                flow=conf.run.flow,
                inputs=parameters,
                callback_to_ui=ui.callback_to_ui,
            )

            # Run the flow synchronously
            logger.info(f"Starting flow '{run_request.flow}'...")

            async def run_with_shutdown() -> None:
                await flow_runner.run_flow(
                    run_request=run_request, wait_for_completion=True
                )
                await bm.graceful_shutdown()

            try:
                asyncio.run(run_with_shutdown())
                logger.info(f"✓ Flow '{run_request.flow}' completed successfully")
            except Exception as e:
                logger.error(f"✗ Flow '{run_request.flow}' failed: {e}")
                raise

        case "batch":
            # Batch mode: Create batch jobs and enqueue them
            flow_name = conf.run.flow
            limit = conf.run.limit
            logger.info(f"Batch mode: Creating jobs for flow '{flow_name}'")
            if limit:
                logger.info(f"Processing limit: {limit} records")

            async def run_with_shutdown() -> None:
                # Get storage config from run section
                storage_config = conf.run.storage_config or None

                await flow_runner.create_batch(
                    flow_name=flow_name,
                    storage_config=storage_config,
                    max_records=limit,
                )
                await bm.graceful_shutdown()

            try:
                asyncio.run(run_with_shutdown())
                logger.info(f"✓ Batch jobs created successfully for flow '{flow_name}'")
            except Exception as e:
                logger.error(f"✗ Batch job creation failed: {e}")
                raise

        case "batch_run":
            # Batch run mode: Process jobs from the queue (worker mode)
            # Each job gets a completely fresh orchestrator instance to ensure
            # no state is shared between jobs, preventing cross-contamination
            # This is critical for research integrity where old state might affect results
            limit = conf.run.limit or 5  # Get limit from config or default to 5
            ui = CLIUserAgent()

            logger.info("Batch run mode: Processing jobs from queue")
            logger.info(f"Maximum jobs to process: {limit}")

            async def run_with_shutdown() -> None:
                summary = await flow_runner.run_batch_job(
                    max_jobs=limit,
                    callback_to_ui=ui.make_callback(),
                    wait_for_completion=True,
                )
                logger.info("\n" + summary.format_for_console())
                await bm.graceful_shutdown()

            try:
                asyncio.run(run_with_shutdown())
                logger.info("✓ Batch processing completed")
            except Exception as e:
                logger.error(f"✗ Batch processing failed: {e}")
                raise

        case "batch_all":
            # Batch all mode: Create and process jobs in one command
            flow_name = conf.run.flow
            limit = conf.run.limit or 999
            logger.info(
                f"Batch all mode: Create and process jobs for flow '{flow_name}'"
            )

            async def run_with_shutdown() -> None:
                # Enqueue phase
                logger.info("Phase 1: Enqueueing batch jobs...")
                storage_config = conf.run.storage_config or None
                await flow_runner.create_batch(
                    flow_name=flow_name,
                    storage_config=storage_config,
                    max_records=limit,
                )
                logger.info("✓ Batch jobs enqueued successfully")

                # Process phase
                ui = CLIUserAgent()
                logger.info(f"Phase 2: Processing batch jobs (limit: {limit})...")
                summary = await flow_runner.run_batch_job(
                    max_jobs=limit,
                    callback_to_ui=ui.make_callback(),
                    wait_for_completion=True,
                )
                logger.info("\n" + summary.format_for_console())

                await bm.graceful_shutdown()

            try:
                asyncio.run(run_with_shutdown())
                logger.info("✓ Batch all mode completed successfully")
            except Exception as e:
                logger.error(f"✗ Batch all mode failed: {e}")
                raise

        case "streamlit":
            # Starts the Streamlit web interface.
            logger.info("Starting Streamlit interface...")
            try:
                from buttermilk.web.streamlit_frontend.app import create_dashboard_app

                # create_dashboard_app is expected to configure and run the Streamlit app.
                # It might need access to flow_runner or specific flow configurations.
                streamlit_app_manager = create_dashboard_app(
                    flow_runner=flow_runner
                )  # Pass FlowRunner
                asyncio.run(
                    streamlit_app_manager.run()
                )  # Assuming create_dashboard_app returns an object with a run method
            except ImportError as e_streamlit:
                logger.error(
                    f"Failed to import Streamlit components: {e_streamlit!s}. Is Streamlit installed?"
                )
            except Exception as e_streamlit_start:
                logger.error(
                    f"Error starting Streamlit interface: {e_streamlit_start!s}",
                    exc_info=True,
                )

        case "api":
            # API mode: Start FastAPI web server for HTTP API access
            host = str(conf.run.host or "0.0.0.0")
            port = int(conf.run.port or 8000)
            logger.info("API mode: Starting FastAPI server")

            # Lazy import FastAPI dependencies only when needed
            import uvicorn  # For running the FastAPI server

            from buttermilk.api.flow import create_app as create_fastapi_app

            # Pass both FlowRunner and the already-initialized BM to avoid re-bootstrapping
            fastapi_app = create_fastapi_app(
                flows=flow_runner,  # Pass the FlowRunner
                bm=bm,  # Pass the already-initialized BM from CLI bootstrap
            )

            # Verify app is ready instead of sleeping
            logger.debug("Verifying FastAPI app readiness...")
            if (
                not hasattr(fastapi_app.state, "flow_runner")
                or not fastapi_app.state.flow_runner
            ):
                raise RuntimeError(
                    "FlowRunner not properly initialized in FastAPI app state"
                )
            logger.debug("✓ FastAPI app readiness verified")

            uvicorn_config = uvicorn.Config(
                app=fastapi_app,
                host=host,
                port=port,
                reload=bool(conf.run.reload or False),  # Hot reloading (dev only)
                log_level=str(conf.run.log_level or "info").lower(),
                access_log=True,  # Enable access logs
                workers=int(conf.run.workers or 1),  # Number of worker processes
                log_config=None,  # Preserve existing logging configuration
            )
            api_server = uvicorn.Server(config=uvicorn_config)
            logger.info(f"✓ Server ready at http://{host}:{port}")
            logger.info(f"   Documentation: http://{host}:{port}/docs")
            try:
                api_server.run()  # This is a blocking call
            except KeyboardInterrupt:
                logger.info("Shutting down gracefully (Ctrl+C received)...")
            finally:
                logger.info("API server stopped.")

        case "pub/sub":
            # Starts a Google Cloud Pub/Sub listener.
            # TODO: Implement Pub/Sub listener functionality
            # This mode was previously delegated to batch_cli, which has been removed.
            # Pub/Sub integration should be implemented using the JobQueueClient with Pub/Sub backend.
            logger.error(
                "Pub/Sub mode is not yet implemented. Use 'batch_run' mode with a Pub/Sub job queue backend instead."
            )
            raise NotImplementedError(
                "Pub/Sub mode requires implementation. See GitHub issues for status."
            )

        case "slackbot":
            # Starts a Slack bot integration.
            logger.info("Starting Slackbot mode...")

            # Retrieve Slack tokens securely from bm.credentials
            slack_creds = bm.credentials
            if not isinstance(slack_creds, dict):
                raise TypeError(
                    f"Expected bm.credentials to be a dict, got {type(slack_creds)}"
                )

            slack_bot_token = slack_creds.get("MODBOT_TOKEN")  # Standard bot token
            slack_app_token = slack_creds.get(
                "SLACK_APP_TOKEN"
            )  # Socket Mode app-level token

            if not slack_bot_token or not slack_app_token:
                raise ValueError(
                    "Missing MODBOT_TOKEN or SLACK_APP_TOKEN in credentials. Check secrets configuration."
                )

            # Set environment variables for Slack Bolt library, if it relies on them.
            # Alternatively, pass tokens directly to initialize_slack_bot if supported.
            os.environ["SLACK_BOT_TOKEN"] = slack_bot_token
            os.environ["SLACK_APP_TOKEN"] = slack_app_token

            from buttermilk.runner.slackbot import (
                initialize_slack_bot,
            )  # Slack bot initialization utility

            event_loop = asyncio.get_event_loop()

            # General functoins might take a few more seconds
            event_loop.slow_callback_duration = 10

            # Queue for managing asyncio tasks created by Slack event handlers
            orchestrator_tasks = asyncio.Queue()  # type: ignore

            slack_bolt_app, slack_bolt_handler = initialize_slack_bot(
                bot_token=slack_bot_token,
                app_token=slack_app_token,
                loop=event_loop,
            )

            # Start the Slack Bolt handler in a background task
            _ = event_loop.create_task(slack_bolt_handler.start_async())

            async def runloop() -> None:
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

        case "pipeline":
            # Pipeline mode: Run data processing pipeline
            logger.info("Pipeline mode: Starting data processing pipeline")

            if not hasattr(conf.run, "pipeline"):
                raise ValueError(
                    "Pipeline configuration missing. Ensure 'run.pipeline' is configured.\n"
                    "Check your pipeline configurations in buttermilk/conf/"
                )

            pipeline_conf = conf.run.pipeline

            # Instantiate source (either via _target_ or as storage config)
            logger.info("Initializing source...")
            source = hydra.utils.instantiate(pipeline_conf["source"])
            # If instantiate didn't create an object (no _target_), treat as storage config
            if isinstance(source, (dict, DictConfig)):
                source = bm.get_storage(source)
            pipeline_conf["source"] = source

            # Instantiate output (either via _target_ or as storage config)
            logger.info("Initializing output...")
            output = hydra.utils.instantiate(pipeline_conf["output"])
            # If instantiate didn't create an object (no _target_), treat as storage config
            if isinstance(output, (dict, DictConfig)):
                output = bm.get_storage(output)
            pipeline_conf["output"] = output

            # Instantiate processors
            logger.info(f"Loading {len(pipeline_conf['processors'])} processor(s)...")
            processors = []
            for proc_conf in pipeline_conf["processors"]:
                processors.append(hydra.utils.instantiate(proc_conf))
            pipeline_conf["processors"] = processors

            # Use run.limit instead of pipeline.max_records for consistency with batch modes
            if conf.run.limit is not None:
                pipeline_conf["limit"] = conf.run.limit
                logger.info(f"Processing limit: {conf.run.limit} records")

            # Instantiate pipeline orchestrator
            from buttermilk.pipeline import PipelineOrchestrator

            orchestrator = PipelineOrchestrator(**pipeline_conf)

            async def run_pipeline() -> None:
                # Ensure tracing is initialized before running pipeline
                # This is required because @weave.op decorators are evaluated at import time
                # but weave.init() hasn't been called yet
                from buttermilk._core.execution_context import get_execution_context

                exec_ctx = get_execution_context()
                await exec_ctx._ensure_tracing_initialized()

                logger.info("Starting pipeline execution...")
                async for _ in orchestrator():
                    pass
                await bm.graceful_shutdown()

            try:
                asyncio.run(run_pipeline())
                logger.info("✓ Pipeline completed successfully")
            except Exception as e:
                logger.error(f"✗ Pipeline failed: {e}")
                raise
        case _:
            # Handles any unsupported modes specified in the configuration.
            valid_modes = [
                "console",
                "batch",
                "batch_run",
                "batch_all",
                "api",
                "pipeline",
                "streamlit",
                "slackbot",
            ]
            raise ValueError(
                f"Unsupported run mode: '{mode}'\n"
                f"Valid modes: {', '.join(valid_modes)}\n"
                f"Usage: python -m buttermilk.runner.cli run.mode=<mode>\n"
                f"Example: python -m buttermilk.runner.cli run.mode=console run.flow=trans"
            )


if __name__ == "__main__":
    # This block executes if the script is run directly (e.g., `python -m buttermilk.runner.cli`).
    # Hydra's `@hydra.main` decorator handles parsing command-line arguments
    # and loading the configuration specified by `config_path` and `config_name`.
    main()
