"""Main command-line interface (CLI) entry point for the Buttermilk application.

This script serves as the primary interface for launching Buttermilk in various
modes. It uses Hydra for configuration management, allowing settings to be
defined in YAML files and overridden via command-line arguments.

Based on the configuration, this script can:
- Run a Buttermilk flow directly in the console for interactive use (`console` mode).
- Start a FastAPI web server to expose Buttermilk flows via an HTTP API (`api` mode).
- Process records through a flow using pipeline (`batch` mode).
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
    flow_required_modes = {"console", "batch"}

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
        - run.mode=<mode>        : Set operational mode (console, batch, api, pipeline, streamlit, slackbot)
        - run.flow=<flow_name>   : Specify which flow to run (required for most modes)
        - run.limit=<number>     : Limit number of records/jobs to process
        - run.record_id=<id>     : Run on a specific record (console mode)
        - run.concurrency=<n>    : Number of concurrent records (batch, pipeline modes)
        - llms=<config>          : Override LLM configuration
        - storage=<config>       : Override storage configuration

    Available Modes:
        - console      : Run a single flow interactively in the terminal
        - batch        : Process records through a flow using pipeline (recommended)
        - api          : Start FastAPI server for HTTP API access
        - pipeline     : Run data processing pipeline
        - streamlit    : Launch Streamlit web interface
        - slackbot     : Start Slack bot integration

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

        case "batch" | "pipeline":
            # Pipeline mode: Run data processing pipeline
            # Batch mode: Generate pipeline config and delegate to pipeline execution
            is_batch_mode = mode == "batch"

            if is_batch_mode:
                # DEPRECATED: Use 'pipeline' mode instead for new projects
                logger.warning(
                    "⚠️  DEPRECATION WARNING: 'batch' mode is deprecated. "
                    "Please migrate to 'pipeline' mode for new projects. "
                    "See RFC #311 for migration guide."
                )

                # Generate pipeline configuration from batch mode parameters
                from buttermilk.processors.unified_processors import GroupchatProcessor
                from buttermilk.utils.utils import expand_dict

                flow_name = conf.run.flow
                flow = flow_runner.flows[flow_name]

                # Get source storage from flow configuration
                if hasattr(flow, "storage") and flow.storage:
                    if "initial" in flow.storage:
                        storage_cfg = flow.storage["initial"]
                    else:
                        storage_cfg = next(iter(flow.storage.values()))
                else:
                    raise ValueError(
                        f"Flow '{flow_name}' has no storage configuration"
                    )

                # Create source storage instance
                source = bm.get_storage(storage_cfg)

                # Expand flow parameters into variants
                param_variants = expand_dict(flow.parameters) if hasattr(flow, 'parameters') and flow.parameters else [{}]

                # Build processors directly for each parameter variant
                processors = []
                for params in param_variants:
                    processors.append(
                        GroupchatProcessor(
                            flow_name=flow_name,
                            flow_config=flow,
                            parameters=params,
                        )
                    )

                # Create generated pipeline configuration
                pipeline_conf = {
                    "pipeline_name": f"batch_{flow_name}",
                    "source": source,
                    "processors": processors,
                    "limit": conf.run.limit,
                    "concurrency": getattr(conf.run, "concurrency", 1) or 1,
                    "enable_record_cache": False,  # Explicit: no caching for orchestrators
                }

                logger.info(
                    f"Batch mode: Generated pipeline config for flow '{flow_name}'"
                )
                if conf.run.limit:
                    logger.info(f"Processing limit: {conf.run.limit} records")
                logger.info(f"Concurrency: {pipeline_conf['concurrency']}")

            else:
                # Pure pipeline mode: load from configuration
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

                if output_cfg := pipeline_conf.get("output", None):
                    # Instantiate output (either via _target_ or as storage config)
                    logger.info("Initializing output...")
                    output = hydra.utils.instantiate(output_cfg)
                    # If instantiate didn't create an object (no _target_), treat as storage config
                    if isinstance(output, (dict, DictConfig)):
                        output = bm.get_storage(output)
                    pipeline_conf["output"] = output

                # Use run.limit instead of pipeline.max_records for consistency
                if conf.run.limit is not None:
                    pipeline_conf["limit"] = conf.run.limit
                    logger.info(f"Processing limit: {conf.run.limit} records")

            # Instantiate processors (pipeline mode only - batch mode already has them)
            if not is_batch_mode:
                logger.info(f"Loading {len(pipeline_conf['processors'])} processor(s)...")
                processors = []

                for proc_conf in pipeline_conf["processors"]:
                    # DictConfig from pipeline mode - use Hydra instantiate
                    processors.append(hydra.utils.instantiate(proc_conf))
                pipeline_conf["processors"] = processors
            else:
                logger.info(f"Using {len(pipeline_conf['processors'])} batch-generated processor(s)...")

            # Instantiate pipeline orchestrator (shared for both modes)
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
                mode_name = "Batch mode" if is_batch_mode else "Pipeline"
                logger.info(f"✓ {mode_name} completed successfully")
            except Exception as e:
                mode_name = "Batch mode" if is_batch_mode else "Pipeline"
                logger.error(f"✗ {mode_name} failed: {e}")
                raise

        case _:
            # Handles any unsupported modes specified in the configuration.
            valid_modes = [
                "console",
                "batch",
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
