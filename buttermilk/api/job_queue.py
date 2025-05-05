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
    _processing: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _subscription_future: Any | None = PrivateAttr(default=None)

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

    async def pull_tasks(self) -> None:
        """Continuously pull tasks from Pub/Sub when the system is idle."""
        if not self.flow_runner:
            logger.warning("JobQueueClient initialized without a FlowRunner. Cannot pull tasks.")
            return

        logger.info(f"Started job worker processing. Listening for messages on {self._jobs_subscription_path}...")
        while True:
            if self.is_system_idle() and not self._processing.is_set():
                try:
                    response = self._subscriber.pull(subscription=self._jobs_subscription_path, max_messages=1, return_immediately=True)
                    if response.received_messages:
                        self._processing.set()
                        message = response.received_messages[0]
                        _ = asyncio.create_task(self._process_message(message))
                except Exception as e:
                    logger.error(f"Error pulling pub/sub message: {e}", exc_info=True)
                    if self._processing.is_set():
                        self._processing.clear()
                    await asyncio.sleep(5)

            await asyncio.sleep(self.idle_check_interval)

    def publish_job(self, job: RunRequest) -> str:
        """Publish a job request to the queue.
        
        Args:
            job: The job definition to publish
            
        Returns:
            The published message ID

        """
        job_data = job.model_dump()
        job_data["type"] = "job_request"
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
        status_data = {
            "type": "job_status",
            "batch_id": batch_id,
            "record_id": record_id,
            "status": status,
        }

        if error:
            status_data["error"] = error

        message_data = json.dumps(status_data).encode("utf-8")
        future = self._publisher.publish(self._status_topic_path, message_data)
        message_id = future.result()

        logger.debug(f"Published status update for job {batch_id}:{record_id}: {status}")
        return message_id

    async def _process_message(self, message: Any) -> None:
        """Parse and dispatch a single message from the Pub/Sub queue."""
        ack_id = message.ack_id
        message_data = message.message.data
        message_obj = message

        try:
            data = json.loads(message_data.decode("utf-8"))

            if data.get("type") != "job_request":
                logger.warning(f"Ignoring message with non-'job_request' type: {data.get('type')}")
                try:
                    self._subscriber.acknowledge(subscription=self._jobs_subscription_path, ack_ids=[ack_id])
                    logger.debug(f"Acked non-job_request message (ack_id: {ack_id})")
                except Exception as ack_err:
                    logger.error(f"Failed to ack non-job message (ack_id: {ack_id}): {ack_err}")
                self._processing.clear()
                return

            if not self.is_system_idle() or self._active_jobs >= self.max_concurrent_jobs:
                logger.info(f"System busy (Idle: {self.is_system_idle()}, Active Jobs: {self._active_jobs}/{self.max_concurrent_jobs}). Not processing job now (ack_id: {ack_id}). Message will be redelivered.")
                self._processing.clear()
                return

            self._active_jobs += 1
            logger.debug(f"Incremented active jobs to {self._active_jobs} for job from message (ack_id: {ack_id})")

            try:
                run_request = RunRequest.model_validate(data)
                if not run_request.job_id:
                    run_request.job_id = ack_id
            except pydantic.ValidationError as e:
                logger.error(f"Validation error creating RunRequest from message (ack_id: {ack_id}): {e}. Message data: {data}")
                try:
                    self._subscriber.acknowledge(subscription=self._jobs_subscription_path, ack_ids=[ack_id])
                    logger.warning(f"Acked malformed message (ack_id: {ack_id})")
                except Exception as ack_err:
                    logger.error(f"Failed to ack malformed message (ack_id: {ack_id}): {ack_err}")

                if self._active_jobs > 0:
                    self._active_jobs -= 1
                self._processing.clear()
                logger.debug(f"Decremented active jobs to {self._active_jobs} after malformed message (ack_id: {ack_id}).")
                return

            logger.info(f"Processing job {run_request.batch_id or 'N/A'}:{run_request.record_id or 'N/A'} (Job ID: {run_request.job_id})")

            loop = asyncio.get_running_loop()
            loop.create_task(self._run_job(run_request, ack_id))

            self.publish_status_update(
                batch_id=run_request.batch_id or "N/A",
                record_id=run_request.record_id or "N/A",
                status=BatchJobStatus.RUNNING,
            )

        except Exception as e:
            logger.error(f"Error processing message envelope (ack_id: {ack_id}): {e}", exc_info=True)
            if self._active_jobs > 0:
                self._active_jobs -= 1
                logger.warning(f"Decremented active jobs to {self._active_jobs} due to top-level processing error (ack_id: {ack_id}).")
            self._processing.clear()
            logger.warning(f"Cleared processing flag due to top-level error (ack_id: {ack_id}).")

    def is_system_idle(self) -> bool:
        """Check if the system is idle (no active user sessions)."""
        from buttermilk.web.activity_tracker import get_instance as get_activity_tracker
        activity_tracker = get_activity_tracker()
        return activity_tracker.is_idle()

    async def _run_job(self, run_request: RunRequest, ack_id: str) -> None:
        """Run a job, update its status, and acknowledge the message upon success."""
        assert self.flow_runner is not None, "FlowRunner must be initialized to run jobs"

        job_desc = f"{run_request.batch_id or 'N/A'}:{run_request.record_id or 'N/A'} (Job ID: {run_request.job_id})"

        try:
            logger.debug(f"Starting flow execution for job {job_desc}")
            result = await self.flow_runner.run_flow(run_request)
            logger.debug(f"Flow execution finished for job {job_desc}. Result type: {type(result)}")

            self.publish_status_update(
                batch_id=run_request.batch_id or "N/A",
                record_id=run_request.record_id or "N/A",
                status=BatchJobStatus.COMPLETED,
            )

            logger.info(f"Job {job_desc} completed successfully")

            try:
                self._subscriber.acknowledge(subscription=self._jobs_subscription_path, ack_ids=[ack_id])
                logger.debug(f"Acknowledged job {job_desc} using ack_id")
            except Exception as ack_error:
                logger.error(
                    f"Failed to acknowledge message for job {job_desc}: {ack_error}",
                    exc_info=True,
                )

        except Exception as e:
            logger.error(f"Error running job {job_desc}: {e}", exc_info=True)

            self.publish_status_update(
                batch_id=run_request.batch_id or "N/A",
                record_id=run_request.record_id or "N/A",
                status=BatchJobStatus.FAILED,
                error=str(e),
            )

        finally:
            if self._active_jobs > 0:
                self._active_jobs -= 1
            else:
                logger.warning(f"Attempted to decrement active_jobs below zero for job {job_desc}. State might be inconsistent.")

            if self._active_jobs == 0:
                self._processing.clear()
                logger.debug("Cleared processing flag as active jobs reached 0.")
            else:
                logger.debug(f"Processing flag remains set. Active jobs: {self._active_jobs}")

            logger.debug(f"Finished _run_job for {job_desc}. Active jobs: {self._active_jobs}")
