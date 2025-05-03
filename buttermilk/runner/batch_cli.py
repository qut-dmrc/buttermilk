"""Command-line interface for batch operations.

This module provides CLI commands for creating and managing batch jobs.
"""

import asyncio
import sys
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from buttermilk._core.batch import BatchErrorPolicy, BatchRequest
from buttermilk.bm import logger
from buttermilk.runner.batch_helper import BatchRunnerHelper
from buttermilk.runner.batchrunner import BatchRunner
from buttermilk.runner.flowrunner import FlowRunner


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for batch CLI.
    
    Allows creating and monitoring batch jobs from the command line.
    """
    
    # Initialize FlowRunner and BatchRunner
    flow_runner = FlowRunner.model_validate(cfg)
    flow_runner.bm.setup_instance()
    
    batch_runner = BatchRunner(flow_runner=flow_runner)
    
    # Get batch configuration
    batch_config = cfg.get("batch", {})
    
    # Get command from arguments or config
    command = batch_config.get("command", "create")
    
    # Handle different commands
    if command == "create":
        asyncio.run(create_batch(batch_runner, cfg))
    elif command == "list":
        asyncio.run(list_batches(batch_runner))
    elif command == "status":
        batch_id = batch_config.get("batch_id")
        if not batch_id:
            logger.error("Batch ID is required for 'status' command")
            sys.exit(1)
        asyncio.run(get_batch_status(batch_runner, batch_id))
    elif command == "cancel":
        batch_id = batch_config.get("batch_id")
        if not batch_id:
            logger.error("Batch ID is required for 'cancel' command")
            sys.exit(1)
        asyncio.run(cancel_batch(batch_runner, batch_id))
    else:
        logger.error(f"Unknown command: {command}")
        sys.exit(1)


async def create_batch(batch_runner: BatchRunner, cfg: DictConfig) -> None:
    """Create a new batch job."""
    flow_name = cfg.flow
    
    # Extract batch parameters from config
    batch_config = cfg.get("batch", {})
    shuffle = batch_config.get("shuffle", True)
    max_records = batch_config.get("max_records", None)
    concurrency = batch_config.get("concurrency", 1)
    error_policy_str = batch_config.get("error_policy", "continue")
    error_policy = (
        BatchErrorPolicy.STOP if error_policy_str.lower() == "stop" else BatchErrorPolicy.CONTINUE
    )
    parameters = batch_config.get("parameters", {})
    interactive = batch_config.get("interactive", False)
    
    logger.info(f"Creating batch job for flow '{flow_name}'")
    logger.info(f"  - shuffle: {shuffle}")
    logger.info(f"  - max_records: {max_records}")
    logger.info(f"  - concurrency: {concurrency}")
    logger.info(f"  - error_policy: {error_policy}")
    logger.info(f"  - interactive: {interactive}")
    
    # Create batch request
    batch_request = BatchRequest(
        flow=flow_name,
        shuffle=shuffle,
        max_records=max_records,
        concurrency=concurrency,
        error_policy=error_policy,
        parameters=parameters,
        interactive=interactive,
    )
    
    try:
        # Create the batch
        batch = await batch_runner.create_batch(batch_request)
        
        logger.info(f"Batch job created successfully: {batch.id}")
        logger.info(f"Total jobs: {batch.total_jobs}")
        
        # Start the worker if requested
        wait = batch_config.get("wait", False)
        if wait:
            logger.info("Waiting for batch to complete...")
            
            # Monitor batch progress
            while True:
                batch = await batch_runner.get_batch_status(batch.id)
                progress = batch.progress * 100
                logger.info(f"Progress: {progress:.1f}% ({batch.completed_jobs}/{batch.total_jobs})")
                
                if batch.status in ['completed', 'failed', 'cancelled']:
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
