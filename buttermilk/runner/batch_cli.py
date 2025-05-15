"""Command-line interface for batch operations.

This module provides CLI commands for creating and managing batch jobs
through the Google Pub/Sub job queue.
"""

import asyncio
import sys

import hydra
from omegaconf import DictConfig

from buttermilk._core.batch import BatchRequest
from buttermilk.bm import BM, logger  # Buttermilk global instance and logger

bm = BM()
from buttermilk.runner.batchrunner import BatchRunner
from buttermilk.runner.flowrunner import FlowRunner


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for batch CLI.
    
    Allows creating and monitoring batch jobs from the command line.
    """
    # Initialize FlowRunner and BatchRunner
    flow_runner = FlowRunner.model_validate(cfg)

    batch_runner = BatchRunner(flow_runner=flow_runner)

    # Get batch configuration
    batch_config = BatchRequest.model_validate({"flow": cfg.flow, **cfg.run.batch})

    # Handle different commands
    if batch_config.command == "create":
        asyncio.run(create_batch(batch_runner, batch_config=batch_config))
    elif batch_config.command == "list":
        asyncio.run(list_batches(batch_runner))
    elif batch_config.command == "status":
        # Get batch ID from parameters
        batch_id = cfg.run.batch.get("batch_id")
        if not batch_id:
            logger.error("Batch ID is required for 'status' command")
            sys.exit(1)
        asyncio.run(get_batch_status(batch_runner, batch_id))
    elif batch_config.command == "cancel":
        # Get batch ID from parameters
        batch_id = cfg.run.batch.get("batch_id")
        if not batch_id:
            logger.error("Batch ID is required for 'cancel' command")
            sys.exit(1)
        asyncio.run(cancel_batch(batch_runner, batch_id))
    else:
        logger.error(f"Unknown command: {batch_config.command}")
        sys.exit(1)


async def create_batch(batch_runner: BatchRunner, batch_config: BatchRequest) -> None:
    """Create a new batch job."""
    try:
        # Create the batch
        batch = await batch_runner.create_batch(batch_config)

        logger.info(f"Batch job created successfully: {batch.id}")
        logger.info(f"Total jobs: {batch.total_jobs}")

        # Start the worker if requested
        wait = batch_config.wait
        if wait:
            logger.info("Waiting for batch to complete...")

            # Monitor batch progress
            while True:
                batch = await batch_runner.get_batch_status(batch.id)
                progress = batch.progress * 100
                logger.info(f"Progress: {progress:.1f}% ({batch.completed_jobs}/{batch.total_jobs})")

                if batch.status in ["completed", "failed", "cancelled"]:
                    logger.info(f"Batch status: {batch.status}")
                    logger.info(f"Completed: {batch.completed_jobs}")
                    logger.info(f"Failed: {batch.failed_jobs}")
                    break

                await asyncio.sleep(5)

    except Exception as e:
        logger.error(f"Error creating batch: {e}")
        sys.exit(1)


async def list_batches(batch_runner: BatchRunner) -> None:
    """List all batches."""
    try:
        # Get all batches
        batches = list(batch_runner._active_batches.values())

        if not batches:
            logger.info("No batches found")
            return

        logger.info(f"Found {len(batches)} batches:")
        for batch in batches:
            progress = batch.progress * 100
            logger.info(f"  - {batch.id}: {batch.flow} - {progress:.1f}% - {batch.status}")

    except Exception as e:
        logger.error(f"Error listing batches: {e}")
        sys.exit(1)


async def get_batch_status(batch_runner: BatchRunner, batch_id: str) -> None:
    """Get status of a batch."""
    try:
        batch = await batch_runner.get_batch_status(batch_id)

        logger.info(f"Batch: {batch.id}")
        logger.info(f"Flow: {batch.flow}")
        logger.info(f"Status: {batch.status}")
        logger.info(f"Created: {batch.created_at}")
        if batch.started_at:
            logger.info(f"Started: {batch.started_at}")
        if batch.completed_at:
            logger.info(f"Completed: {batch.completed_at}")

        progress = batch.progress * 100
        logger.info(f"Progress: {progress:.1f}%")
        logger.info(f"Jobs: {batch.completed_jobs}/{batch.total_jobs} completed, {batch.failed_jobs} failed")

    except ValueError:
        logger.error(f"Batch '{batch_id}' not found")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error getting batch status: {e}")
        sys.exit(1)


async def cancel_batch(batch_runner: BatchRunner, batch_id: str) -> None:
    """Cancel a batch."""
    try:
        batch = await batch_runner.cancel_batch(batch_id)
        logger.info(f"Batch '{batch_id}' cancelled")

    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error cancelling batch: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # This block executes when the script is run directly
    main()
