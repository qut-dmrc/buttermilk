"""
Unified Buttermilk CLI using Typer.

Simple top-level commands for all Buttermilk modes:
    bm batch <flow>    # Run batch workflows
    bm console <flow>  # Run a single flow (console mode)
    bm api             # Start API server
    bm pipeline        # Run pipeline

This wraps the existing infrastructure with a simpler, more intuitive interface.
"""
import asyncio
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from typing_extensions import Annotated

from buttermilk import init_async, logger
from buttermilk._core.types import RunRequest
from buttermilk.agents.ui.console import CLIUserAgent
from buttermilk.api.flow import create_app as create_fastapi_app
from buttermilk.runner.flowrunner import FlowRunner

app = typer.Typer(help="Buttermilk: LLM workflows for research")


@app.command("batch", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def batch(
    ctx: typer.Context,
    flow: Annotated[str, typer.Argument(help="Name of the flow to run")],
    enqueue_only: Annotated[bool, typer.Option("--enqueue-only", help="Only enqueue jobs, don't process")] = False,
    process_only: Annotated[bool, typer.Option("--process-only", help="Only process jobs, don't enqueue")] = False,
    max_records: Annotated[Optional[int], typer.Option("--max-records", help="Maximum number of records to process")] = None,
    max_jobs: Annotated[Optional[int], typer.Option("--max-jobs", help="Maximum number of jobs to process")] = None,
    config_dir: Annotated[Optional[str], typer.Option("--config-dir", help="Path to config directory")] = None,
):
    """
    Run a batch workflow: enqueue and/or process jobs.

    Supports Hydra-style overrides for config composition:
        bm batch trans flows=trans llms=flash
        bm batch trans flows=[trans,judge] storage=tja

    Examples:
        bm batch trans                    # Enqueue and process all
        bm batch trans --enqueue-only     # Only enqueue
        bm batch trans flows=trans        # With Hydra override
        bm batch trans --max-records 100  # Limit to 100 records
    """
    # Validate mutually exclusive options
    if enqueue_only and process_only:
        typer.echo("Error: --enqueue-only and --process-only are mutually exclusive. Cannot use both.", err=True)
        raise typer.Exit(1)

    # Determine mode
    if enqueue_only:
        mode = "enqueue"
    elif process_only:
        mode = "process"
    else:
        mode = "all"

    # Extract Hydra overrides from extra args (e.g., flows=trans, llms=flash)
    hydra_overrides = ctx.args if ctx.args else []

    # Run async batch operation
    try:
        asyncio.run(run_batch_async(
            flow_name=flow,
            mode=mode,
            max_records=max_records,
            max_jobs=max_jobs,
            config_dir=config_dir,
            overrides=hydra_overrides
        ))
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Fatal error: {e}", err=True)
        logger.exception("Batch processing failed")
        raise typer.Exit(1)


@app.command("console")
def console(
    flow: Annotated[str, typer.Argument(help="Name of the flow to run")],
    record_id: Annotated[Optional[str], typer.Option("--record-id", help="Record ID to process")] = None,
    prompt: Annotated[Optional[str], typer.Option("--prompt", help="Prompt to use")] = None,
    uri: Annotated[Optional[str], typer.Option("--uri", help="URI to fetch")] = None,
    config_dir: Annotated[Optional[str], typer.Option("--config-dir", help="Path to config directory")] = None,
):
    """
    Run a single flow in console mode.

    Examples:
        bm console trans                          # Run flow 'trans'
        bm console trans --record-id 123          # Run with specific record
        bm console trans --prompt "analyze this"  # Run with prompt
    """
    try:
        asyncio.run(run_flow_console_async(
            flow_name=flow,
            record_id=record_id,
            prompt=prompt,
            uri=uri,
            config_dir=config_dir
        ))
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Fatal error: {e}", err=True)
        logger.exception("Flow execution failed")
        raise typer.Exit(1)


@app.command()
def api(
    host: Annotated[str, typer.Option("--host", help="Host to bind to")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", help="Port to listen on")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Enable auto-reload")] = False,
    workers: Annotated[int, typer.Option("--workers", help="Number of worker processes")] = 1,
    config_dir: Annotated[Optional[str], typer.Option("--config-dir", help="Path to config directory")] = None,
):
    """
    Start the FastAPI server.

    Examples:
        bm api                    # Start on default port 8000
        bm api --port 8080        # Start on port 8080
        bm api --reload           # Start with auto-reload
    """
    try:
        asyncio.run(run_api_async(
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            config_dir=config_dir
        ))
    except Exception as e:
        typer.echo(f"Fatal error starting API server: {e}", err=True)
        logger.exception("API server failed to start")
        raise typer.Exit(1)


@app.command()
def pipeline(
    config_dir: Annotated[Optional[str], typer.Option("--config-dir", help="Path to config directory")] = None,
):
    """
    Run a pipeline workflow.

    Pipelines are configured in your config files under the 'pipeline' section.

    Examples:
        bm pipeline                        # Run pipeline from config
        bm pipeline --config-dir /path     # Use custom config
    """
    try:
        asyncio.run(run_pipeline_async(config_dir=config_dir))
    except Exception as e:
        typer.echo(f"Fatal error running pipeline: {e}", err=True)
        logger.exception("Pipeline execution failed")
        raise typer.Exit(1)


async def run_batch_async(
    flow_name: str,
    mode: str = "all",
    max_records: Optional[int] = None,
    max_jobs: Optional[int] = None,
    config_dir: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> None:
    """
    Execute batch processing asynchronously with Hydra composition support.

    Args:
        flow_name: Name of the flow to run
        mode: Operation mode - 'all', 'enqueue', or 'process'
        max_records: Maximum records to enqueue
        max_jobs: Maximum jobs to process
        config_dir: Optional config directory path
        overrides: Hydra-style config overrides (e.g., ['flows=trans', 'llms=flash'])
    """
    # Initialize Buttermilk with Hydra composition
    init_kwargs = {
        "job": f"batch_{flow_name}",
        "project_name": "batch",
    }

    if config_dir:
        init_kwargs["config_dir"] = config_dir

    if overrides:
        init_kwargs["overrides"] = overrides

    bm = await init_async(**init_kwargs)

    try:
        # Create FlowRunner with flows from config
        flow_runner = FlowRunner(flows=bm.cfg.flows)
        flow_runner.set_session_bm(bm)

        # Validate flow exists
        if flow_name not in bm.cfg.flows:
            available = list(bm.cfg.flows.keys())
            raise ValueError(
                f"Flow '{flow_name}' not found. Available flows: {available}"
            )

        # Execute based on mode
        if mode in ["all", "enqueue"]:
            logger.info(f"Enqueueing batch jobs for flow '{flow_name}'...")
            await flow_runner.create_batch(
                flow_name=flow_name,
                storage_config=None,  # Auto-discover from flow config
                max_records=max_records
            )
            logger.info("Batch jobs enqueued successfully")

        if mode in ["all", "process"]:
            ui = CLIUserAgent()
            jobs_to_process = max_jobs or 999  # Process all if not specified

            logger.info(f"Processing batch jobs (max: {jobs_to_process})...")
            await flow_runner.run_batch_job(
                callback_to_ui=ui.callback_to_ui,
                max_jobs=jobs_to_process,
                wait_for_completion=True
            )
            logger.info("Batch processing completed successfully")

    finally:
        # Always cleanup
        await bm.graceful_shutdown()


async def run_flow_console_async(
    flow_name: str,
    record_id: Optional[str] = None,
    prompt: Optional[str] = None,
    uri: Optional[str] = None,
    config_dir: Optional[str] = None,
) -> None:
    """
    Execute a single flow in console mode asynchronously.

    Args:
        flow_name: Name of the flow to run
        record_id: Optional record ID to process
        prompt: Optional prompt to use
        uri: Optional URI to fetch
        config_dir: Optional config directory path
    """
    # Initialize Buttermilk
    init_kwargs = {
        "job": f"console_{flow_name}",
        "project_name": "console",
    }

    if config_dir:
        init_kwargs["config_dir"] = config_dir

    bm = await init_async(**init_kwargs)

    try:
        # Create FlowRunner
        flow_runner = FlowRunner(flows=bm.cfg.flows)
        flow_runner.set_session_bm(bm)

        # Validate flow exists
        if flow_name not in bm.cfg.flows:
            available = list(bm.cfg.flows.keys())
            raise ValueError(
                f"Flow '{flow_name}' not found. Available flows: {available}"
            )

        # Build parameters
        parameters = {}
        if record_id:
            parameters["record_id"] = record_id
        if prompt:
            parameters["prompt"] = prompt
        if uri:
            parameters["uri"] = uri

        # Create UI callback
        ui = CLIUserAgent()

        # Create run request
        run_request = RunRequest(
            flow=flow_name,
            inputs=parameters,
            callback_to_ui=ui.callback_to_ui,
        )

        # Execute flow
        logger.info(f"Running flow '{flow_name}' in console mode...")
        await flow_runner.run_flow(run_request=run_request, wait_for_completion=True)
        logger.info(f"Flow '{flow_name}' completed successfully")

    finally:
        await bm.graceful_shutdown()


async def run_api_async(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
    config_dir: Optional[str] = None,
) -> None:
    """
    Start the FastAPI server asynchronously.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload
        workers: Number of worker processes
        config_dir: Optional config directory path
    """
    # Initialize Buttermilk
    init_kwargs = {
        "job": "api_server",
        "project_name": "api",
    }

    if config_dir:
        init_kwargs["config_dir"] = config_dir

    bm = await init_async(**init_kwargs)

    try:
        # Create FlowRunner
        flow_runner = FlowRunner(flows=bm.cfg.flows)
        flow_runner.set_session_bm(bm)

        # Create FastAPI app
        fastapi_app = create_fastapi_app(flows=flow_runner, bm=bm)

        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app=fastapi_app,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True,
            workers=workers,
            log_config=None,
        )

        api_server = uvicorn.Server(config=uvicorn_config)
        logger.info(f"FastAPI server starting on http://{host}:{port}")

        try:
            await api_server.serve()
        except KeyboardInterrupt:
            logger.info("FastAPI server shutting down...")

    finally:
        await bm.graceful_shutdown()


async def run_pipeline_async(config_dir: Optional[str] = None) -> None:
    """
    Execute a pipeline workflow asynchronously.

    Args:
        config_dir: Optional config directory path
    """
    # Initialize Buttermilk
    init_kwargs = {
        "job": "pipeline",
        "project_name": "pipeline",
    }

    if config_dir:
        init_kwargs["config_dir"] = config_dir

    bm = await init_async(**init_kwargs)

    try:
        # Import pipeline components
        from buttermilk.pipeline import PipelineOrchestrator
        from buttermilk.tools.catalog_test import TMDBTool
        from buttermilk.utils.uploader import AsyncDataUploader

        # Get pipeline configuration
        pipeline_conf = bm.cfg.get("pipeline", {})
        if not pipeline_conf:
            raise ValueError("No pipeline configuration found. Add 'pipeline' section to your config.")

        # Set up data source
        source_config = pipeline_conf.get("source")
        if not source_config:
            raise ValueError("Pipeline mode requires 'pipeline.source' configuration")

        # Get storage for source
        source_storage = bm.get_storage(source_config)

        # Set up processors
        processors = []

        # Add TMDB processor if configured
        tmdb_conf = pipeline_conf.get("tmdb", {})
        if isinstance(tmdb_conf, bool):
            tmdb_conf = {} if tmdb_conf else None
        if tmdb_conf is not None:
            tmdb_tool = TMDBTool(**tmdb_conf)
            processors.append(tmdb_tool)
            logger.info(f"Added TMDB processor with region={tmdb_tool.region}")

        # Add uploader processor
        output_storage = bm.get_storage(pipeline_conf.get("output"))
        uploader = AsyncDataUploader(
            storage=output_storage,
            buffer_size=pipeline_conf.get("buffer_size", 10),
            flush_interval=pipeline_conf.get("flush_interval", 30)
        )
        processors.append(uploader)
        logger.info(f"Added uploader with buffer_size={uploader.buffer_size}")

        # Create orchestrator
        concurrency = pipeline_conf.get("concurrency", 1)
        max_records = pipeline_conf.get("max_records")

        orchestrator = PipelineOrchestrator(
            stage_name="pipeline",
            concurrency=concurrency,
            max_records=max_records,
            source=source_storage,
            processors=processors,
        )

        logger.info(f"Running pipeline with {len(processors)} processors (concurrency={concurrency})...")

        # Process pipeline
        async for _ in orchestrator():
            pass  # Processing happens inside orchestrator

        # Cleanup processors
        for proc in processors:
            if hasattr(proc, "shutdown"):
                proc.shutdown()

        logger.info("Pipeline processing complete")

    finally:
        await bm.graceful_shutdown()


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
