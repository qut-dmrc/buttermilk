"""Job queue client for interacting with Google Pub/Sub.

This module provides a client for publishing and subscribing to job messages
via Google Pub/Sub, allowing for decoupled job creation and execution. It
also contains the daemon that monitors the Pub/Sub queue for jobs
and processes them when the system is idle (no interactive API or websocket activity).
"""

import asyncio
import json
from typing import Any, Self

import pydantic
from google.cloud.pubsub import PublisherClient, SubscriberClient
from pydantic import BaseModel, Field, PrivateAttr

from buttermilk._core.batch import BatchJobStatus
from buttermilk._core.types import RunRequest
from buttermilk.bm import bm, logger
from buttermilk.runner.flowrunner import FlowRunner
from buttermilk.web.activity_tracker import get_instance as get_activity_tracker


class JobQueueClient(BaseModel):
    """Client for interacting with the Google Pub/Sub job queue.
    
    This class provides methods for:
    - Publishing job requests to the queue
    - Publishing job status updates
    - Checking if the system is idle
    """

    flow_runner: FlowRunner | None = Field(default=None)
    idle_check_interval: float = 5.0
    max_concurrent_jobs: int = 1

    _active_jobs: int = PrivateAttr(default=0)
    _publisher: PublisherClient = PrivateAttr(default_factory=PublisherClient)
    _subscriber: SubscriberClient = PrivateAttr(default_factory=SubscriberClient)
    _jobs_subscription_path: str = PrivateAttr()
    _status_subscription_path: str = PrivateAttr()
    _status_topic_path: str = PrivateAttr()
    _jobs_topic_path: str = PrivateAttr()
    _subscription_future: asyncio.Future | None = PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def _setup(self) -> Self:
        self._jobs_subscription_path = self._subscriber.subscription_path(
            bm.pubsub.project,
            bm.pubsub.jobs_subscription,
        )
        self._status_subscription_path = self._subscriber.subscription_path(
            bm.pubsub.project,
            bm.pubsub.status_subscription,
        )
        self._status_topic_path = self._subscriber.topic_path(bm.pubsub.project, bm.pubsub.status_topic)
        self._jobs_topic_path = self._subscriber.topic_path(bm.pubsub.project, bm.pubsub.jobs_topic)

        return self

    async def pull_tasks(self):
        # If we were initialised with a runner, set up the callback to process messages
        if not self.flow_runner:
            return

        logger.info(f"Started job worker processing. Listening for messages with {self._jobs_subscription_path} topic {self._jobs_topic_path}...")
        while True:
            if self.is_system_idle() and not self._processing.is_set():
                try:
                    self._processing.set()
                    if messages := self._subscriber.pull(subscription=self._jobs_subscription_path, max_messages=1):
                        for job in messages.received_messages:
                            task = asyncio.create_task(self._process_message(job))
                except Exception as e:
                    self._processing.clear()
                    logger.error(f"Error pulling pub/sub message: {e}")

            await asyncio.sleep(1)

    def publish_job(self, job: RunRequest) -> str:
        """Publish a job request to the queue.
        
        Args:
            job: The job definition to publish
            
        Returns:
            The published message ID

        """
        # Convert job to JSON
        job_data = job.model_dump()

        # Add the job type for routing
        job_data["type"] = "job_request"

        # Publish to the jobs topic
        message_data = json.dumps(job_data).encode("utf-8")
        future = self._publisher.publish(self._jobs_topic_path, message_data)
        message_id = future.result()

        logger.info(f"Published job {job.batch_id}:{job.record_id} to Pub/Sub (ID: {message_id}, topic: {self._jobs_topic_path})")
        return message_id

    def publish_status_update(self,
                              batch_id: str,
                              record_id: str,
                              status: BatchJobStatus,
                              error: str | None = None) -> str:
        """Publish a job status update.
        
        Args:
            batch_id: The batch ID
            record_id: The record ID for the job
            status: The current job status
            error: Optional error message if the job failed
            
        Returns:
            The published message ID

        """
        # Create status update data
        status_data = {
            "type": "job_status",
            "batch_id": batch_id,
            "record_id": record_id,
            "status": status,
        }

        if error:
            status_data["error"] = error

        # Publish to the status topic
        message_data = json.dumps(status_data).encode("utf-8")
        future = self._publisher.publish(self._status_topic_path, message_data)
        message_id = future.result()

        logger.debug(f"Published status update for job {batch_id}:{record_id}: {status}")
        return message_id

    async def _process_message(self, message):
        """Process a message from the Pub/Sub queue.
        
        Args:
            message: The Pub/Sub message containing the job details

        """
        ack_id = None
        try:
            # Parse the message data
            if hasattr(message, "message"):
                ack_id = message.ack_id
                message = message.message
            data = json.loads(message.data.decode("utf-8"))

            # Only process job requests
            if data.get("type") != "job_request":
                logger.warning(f"Ignoring message with type {data.get('type')}")
                if ack_id:
                    self._subscriber.acknowledge(ack_ids=[ack_id])
                else:
                    message.ack()
                return

            # Check if we can take on another job
            if not self.is_system_idle() or self._active_jobs >= self.max_concurrent_jobs:
                # We're not idle or at max capacity, nack the message so it gets redelivered
                logger.debug("System busy, not processing job now")
                if not ack_id:
                    message.nack()
                return

            # Mark the system as processing
            self._processing.set()
            self._active_jobs += 1

            # Create a RunRequest object from the message data
            try:
                run_request = RunRequest.model_validate(data)

            except Exception as e:
                logger.error(f"Error creating RunRequest from message: {e}")
                # message.ack()  # Don't retry if the message is malformed
                self._processing.clear()
                self._active_jobs -= 1

            logger.info(f"Processing job {run_request.batch_id}:{run_request.record_id}")

            # Run the job asynchronously
            loop = asyncio.get_running_loop()
            loop.create_task(self._run_job(run_request, message))
            # Publish a status update: job is running
            self.publish_status_update(
                batch_id=run_request.batch_id,
                record_id=run_request.record_id,
                status=BatchJobStatus.RUNNING,
            )
            if ack_id:
                self._subscriber.acknowledge(subscription=self._jobs_subscription_path, ack_ids=[ack_id])
            else:
                message.ack()

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            if not ack_id:
                message.nack()  # Retry the message

    def stop(self):
        """Stop the job worker daemon."""
        if not self.running:
            logger.warning("Job worker is not running")
            return

        logger.info("Stopping job worker daemon")
        self.running = False

        if self._subscription_future:
            self._subscription_future.cancel()

        if self._subscriber:
            self._subscriber.close()

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
        return activity_tracker.is_idle()

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
            self.publish_status_update(
                batch_id=run_request.batch_id,
                record_id=run_request.record_id,
                status=BatchJobStatus.COMPLETED,
            )

            logger.info(f"Job {run_request.batch_id}:{run_request.record_id} completed successfully")

        except Exception as e:
            logger.error(f"Error running job {run_request.batch_id}:{run_request.record_id}: {e}")

            # Publish failure status update
            self.publish_status_update(
                batch_id=run_request.batch_id,
                record_id=run_request.record_id,
                status=BatchJobStatus.FAILED,
                error=str(e),
            )
