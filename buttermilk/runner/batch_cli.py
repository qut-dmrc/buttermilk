"""
Simplified batch CLI using Typer.

Provides a user-friendly command-line interface for batch processing:
    bm run batch <flow> [options]

This wraps the existing Hydra-based infrastructure with a simpler interface.
"""
import asyncio
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from buttermilk import init_async, logger
from buttermilk.agents.ui.console import CLIUserAgent
from buttermilk.runner.flowrunner import FlowRunner

app = typer.Typer(help="Buttermilk: LLM workflows for research")


def run_command() -> typer.Typer:
    """Create the 'run' subcommand group."""
    run_app = typer.Typer(help="Run Buttermilk workflows")

    @run_app.command("batch")
    def batch(
        flow: Annotated[str, typer.Argument(help="Name of the flow to run")],
        enqueue_only: Annotated[bool, typer.Option("--enqueue-only", help="Only enqueue jobs, don't process")] = False,
        process_only: Annotated[bool, typer.Option("--process-only", help="Only process jobs, don't enqueue")] = False,
        max_records: Annotated[Optional[int], typer.Option("--max-records", help="Maximum number of records to process")] = None,
        max_jobs: Annotated[Optional[int], typer.Option("--max-jobs", help="Maximum number of jobs to process")] = None,
        config_dir: Annotated[Optional[str], typer.Option("--config-dir", help="Path to config directory")] = None,
    ):
        """
        Run a batch workflow: enqueue and/or process jobs.

        Examples:
            bm run batch trans                    # Enqueue and process all
            bm run batch trans --enqueue-only     # Only enqueue
            bm run batch trans --process-only     # Only process
            bm run batch trans --max-records 100  # Limit to 100 records
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

        # Run async batch operation
        try:
            asyncio.run(run_batch_async(
                flow_name=flow,
                mode=mode,
                max_records=max_records,
                max_jobs=max_jobs,
                config_dir=config_dir
            ))
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"Fatal error: {e}", err=True)
            logger.exception("Batch processing failed")
            raise typer.Exit(1)

    return run_app


# Register the run command group
app.add_typer(run_command(), name="run")


async def run_batch_async(
    flow_name: str,
    mode: str = "all",
    max_records: Optional[int] = None,
    max_jobs: Optional[int] = None,
    config_dir: Optional[str] = None,
) -> None:
    """
    Execute batch processing asynchronously.

    Args:
        flow_name: Name of the flow to run
        mode: Operation mode - 'all', 'enqueue', or 'process'
        max_records: Maximum records to enqueue
        max_jobs: Maximum jobs to process
        config_dir: Optional config directory path
    """
    # Initialize Buttermilk
    init_kwargs = {
        "job": f"batch_{flow_name}",
        "project_name": "batch",
    }

    if config_dir:
        init_kwargs["config_dir"] = config_dir

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


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
