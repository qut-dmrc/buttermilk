"""Job worker daemon for processing jobs from the Pub/Sub queue.

This module contains the daemon that monitors the Pub/Sub queue for jobs
and processes them when the system is idle (no interactive API or websocket activity).
"""

import asyncio
import json

from google.cloud import pubsub_v1

from buttermilk._core.batch import BatchJobStatus
from buttermilk._core.types import RunRequest
from buttermilk.api.job_queue import JobQueueClient
from buttermilk.bm import logger
from buttermilk.runner.flowrunner import FlowRunner
from buttermilk.web.activity_tracker import get_instance as get_activity_tracker


class JobWorker:
    """Worker daemon for processing jobs from the Pub/Sub queue.
    
    This class:
    - Subscribes to the Pub/Sub job queue
    - Monitors system activity
    - Processes jobs when the system is idle
    """

    def __init__(self,
                 flow_runner: FlowRunner,
                 job_queue: JobQueueClient | None = None,
                 idle_check_interval: float = 5.0,
                 max_concurrent_jobs: int = 1):
        """Initialize the job worker.
        
        Args:
            flow_runner: The flow runner to use for job execution
            job_queue: Optional job queue client (created if not provided)
            idle_check_interval: How often to check if the system is idle (seconds)
            max_concurrent_jobs: Maximum number of jobs to process at once

        """
        self.flow_runner = flow_runner
        self.idle_check_interval = idle_check_interval
        self.max_concurrent_jobs = max_concurrent_jobs
        self.running = False
        self.active_jobs = 0
        self.subscriber = None
        self.subscription_future = None

        # Use provided job queue or create a new one
        if job_queue:
            self.job_queue = job_queue
        else:
            try:
                self.job_queue = JobQueueClient()
                logger.info("Created job queue client for worker")
            except Exception as e:
                logger.error(f"Failed to create job queue client: {e}")
                raise

    async def start(self):
        """Start the job worker daemon."""
        if self.running:
            logger.warning("Job worker is already running")
            return

        self.running = True
        logger.info("Starting job worker daemon")

        try:
            # Create subscriber client
            self.subscriber = pubsub_v1.SubscriberClient()
            # Use the subscription path from the job queue
            try:
                # Use values from bm.cfg if available
                from buttermilk import bm
                project_id = getattr(bm, "cfg", {}).get("pubsub", {}).get("project", "buttermilk-project")
                jobs_topic = getattr(bm, "cfg", {}).get("pubsub", {}).get("jobs_topic", "buttermilk-jobs")
                subscription = getattr(bm, "cfg", {}).get("pubsub", {}).get("subscription", f"{jobs_topic}-sub")

                subscription_path = self.subscriber.subscription_path(
                    project_id,
                    subscription,
                )
            except (AttributeError, ImportError) as e:
                logger.warning(f"Could not get Pub/Sub configuration from bm.cfg: {e}")
                # Use default values or raise an error
                raise ValueError("Pub/Sub configuration not available")

            # Set up the callback to process messages
            self.subscription_future = self.subscriber.subscribe(
                subscription_path,
                callback=self._process_message,
            )

            logger.info(f"Subscribed to job queue: {subscription_path}")

            # Start the idle monitoring loop
            while self.running:
                if self.is_system_idle() and self.active_jobs < self.max_concurrent_jobs:
                    # The system is idle and we can process more jobs
                    # Jobs will be pulled via the subscription callback
                    pass

                # Sleep before checking again
                await asyncio.sleep(self.idle_check_interval)

        except Exception as e:
            logger.error(f"Error in job worker: {e}")
            self.running = False
            if self.subscription_future:
                self.subscription_future.cancel()
            if self.subscriber:
                self.subscriber.close()

    def stop(self):
        """Stop the job worker daemon."""
        if not self.running:
            logger.warning("Job worker is not running")
            return

        logger.info("Stopping job worker daemon")
        self.running = False

        if self.subscription_future:
            self.subscription_future.cancel()

        if self.subscriber:
            self.subscriber.close()

    def is_system_idle(self) -> bool:
        """Check if the system is idle and can process jobs.
        
        Checks both the ActivityTracker to see if there are any active
        user sessions, and the job queue's own idle status.
        
        Returns:
            True if the system is idle, False otherwise

        """
        # Get activity tracker instance
        activity_tracker = get_activity_tracker()

        # System is idle if both:
        # 1. The activity tracker says it's idle (no active websockets/API requests)
        # 2. The job queue itself is not processing anything
        return activity_tracker.is_idle() and self.job_queue.is_idle()

    def _process_message(self, message):
        """Process a message from the Pub/Sub queue.
        
        Args:
            message: The Pub/Sub message containing the job details

        """
        try:
            # Parse the message data
            data = json.loads(message.data.decode("utf-8"))

            # Only process job requests
            if data.get("type") != "job_request":
                logger.warning(f"Ignoring message with type {data.get('type')}")
                message.ack()
                return

            # Check if we can take on another job
            if not self.is_system_idle() or self.active_jobs >= self.max_concurrent_jobs:
                # We're not idle or at max capacity, nack the message so it gets redelivered
                logger.debug("System busy, not processing job now")
                message.nack()
                return

            # Mark the system as processing
            self.job_queue.set_processing_status(True)
            self.active_jobs += 1

            # Create a RunRequest object from the message data
            try:
                # Create a request directly - we no longer use BatchJobDefinition
                run_request = RunRequest.model_validate(data)

                # Publish a status update: job is running
                self.job_queue.publish_status_update(
                    batch_id=run_request.batch_id,
                    record_id=run_request.record_id,
                    status=BatchJobStatus.RUNNING,
                )

                logger.info(f"Processing job {run_request.batch_id}:{run_request.record_id}")

                # Run the job asynchronously
                asyncio.create_task(self._run_job(run_request, message))

            except Exception as e:
                logger.error(f"Error creating RunRequest from message: {e}")
                message.ack()  # Don't retry if the message is malformed
                self.job_queue.set_processing_status(False)
                self.active_jobs -= 1

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            message.nack()  # Retry the message

    async def _run_job(self, run_request: RunRequest, message):
        """Run a job and update its status.
        
        Args:
            run_request: The RunRequest to execute
            message: The original Pub/Sub message

        """
        try:
            # Run the flow directly with the request
            result = await self.flow_runner.run_flow(run_request)

            # Publish success status update
            self.job_queue.publish_status_update(
                batch_id=run_request.batch_id,
                record_id=run_request.record_id,
                status=BatchJobStatus.COMPLETED,
            )

            logger.info(f"Job {run_request.batch_id}:{run_request.record_id} completed successfully")
            message.ack()

        except Exception as e:
            logger.error(f"Error running job {run_request.batch_id}:{run_request.record_id}: {e}")

            # Publish failure status update
            self.job_queue.publish_status_update(
                batch_id=run_request.batch_id,
                record_id=run_request.record_id,
                status=BatchJobStatus.FAILED,
                error=str(e),
            )

            message.ack()  # Still ack the message, but mark it as failed

        finally:
            # Mark the system as not processing
            self.job_queue.set_processing_status(False)
            self.active_jobs -= 1


# Function to create and start a job worker
async def start_job_worker(flow_runner: FlowRunner,
                          job_queue: JobQueueClient | None = None,
                          max_concurrent_jobs: int = 1) -> JobWorker:
    """Create and start a job worker.
    
    Args:
        flow_runner: The flow runner to use for job execution
        job_queue: Optional job queue client (created if not provided)
        max_concurrent_jobs: Maximum number of jobs to process at once
        
    Returns:
        The started job worker

    """
    worker = JobWorker(
        flow_runner=flow_runner,
        job_queue=job_queue,
        max_concurrent_jobs=max_concurrent_jobs,
    )

    # Start the worker in a background task
    asyncio.create_task(worker.start())

    return worker
