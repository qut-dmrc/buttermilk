"""API endpoints for batch operations.

This module contains the FastAPI endpoints for managing batch jobs, including
creating batches, checking status, and retrieving results.
"""


from fastapi import APIRouter, HTTPException, Path, Query

from buttermilk._core.batch import BatchMetadata, BatchRequest
from buttermilk.bm import BM, logger  # Buttermilk global instance and logger

bm = BM()
from buttermilk.runner.batchrunner import BatchRunner


def create_batch_router(batch_runner: BatchRunner) -> APIRouter:
    """Create a router for batch endpoints.
    
    Args:
        batch_runner: The BatchRunner instance to use for batch operations
        
    Returns:
        An APIRouter with batch endpoints

    """
    router = APIRouter()

    @router.post("/batches", response_model=BatchMetadata, tags=["batches"])
    async def create_batch(batch_request: BatchRequest):
        """Create a new batch job."""
        try:
            batch_metadata = await batch_runner.create_batch(batch_request)
            return batch_metadata
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error creating batch: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    @router.get("/batches", tags=["batches"])
    async def list_batches(
        status: str | None = Query(None, description="Filter by status"),
    ):
        """List all batches."""
        try:
            # This is a simplified approach - in a real implementation,
            # you'd probably want pagination and more sophisticated filtering
            batches = list(batch_runner._active_batches.values())

            # Apply status filter if provided
            if status:
                batches = [b for b in batches if b.status == status]

            return batches
        except Exception as e:
            logger.error(f"Error listing batches: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    @router.get("/batches/{batch_id}", response_model=BatchMetadata, tags=["batches"])
    async def get_batch(
        batch_id: str = Path(..., description="The ID of the batch to retrieve"),
    ):
        """Get batch status."""
        try:
            return await batch_runner.get_batch_status(batch_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error getting batch {batch_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    @router.delete("/batches/{batch_id}", response_model=BatchMetadata, tags=["batches"])
    async def cancel_batch(
        batch_id: str = Path(..., description="The ID of the batch to cancel"),
    ):
        """Cancel a batch job."""
        try:
            return await batch_runner.cancel_batch(batch_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error cancelling batch {batch_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    @router.get("/batches/{batch_id}/jobs", tags=["batches"])
    async def get_batch_jobs(
        batch_id: str = Path(..., description="The ID of the batch"),
    ):
        """Get all jobs for a batch."""
        try:
            return await batch_runner.get_batch_jobs(batch_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error getting jobs for batch {batch_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    @router.get("/batches/{batch_id}/jobs/{record_id}", tags=["batches"])
    async def get_job_result(
        batch_id: str = Path(..., description="The ID of the batch"),
        record_id: str = Path(..., description="The record ID of the job"),
    ):
        """Get the result of a specific job."""
        try:
            return await batch_runner.get_job_result(batch_id, record_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error getting result for job {record_id} in batch {batch_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    return router
