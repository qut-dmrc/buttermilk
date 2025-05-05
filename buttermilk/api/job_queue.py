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

# Use the async subscriber client
from google.cloud.pubsub_v1.publisher.client import Client as PublisherClient
from google.cloud.pubsub_v1.subscriber.client import Client as SubscriberClient
from google.cloud.pubsub_v1.subscriber.message import Message

# Import FlowControl
from google.cloud.pubsub_v1.types import FlowControl
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

    async def pull_tasks(self):
        """Starts the subscriber to listen for tasks using streaming pull."""
        if not self.flow_runner:
            return

        # Delay until the rest of the system is ready
        await asyncio.sleep(10)

        # Configure FlowControl to only handle one message at a time
        flow_control = FlowControl(max_messages=1)

        logger.info(f"Starting job worker. Listening on {self._jobs_subscription_path}...")

        try:
            # Start the subscription
            self._subscription_future = self._subscriber.subscribe(
                self._jobs_subscription_path,
                callback=self._process_message_callback,
                flow_control=flow_control,
            )

            logger.info(f"Subscriber started for {self._jobs_subscription_path}.")
        except asyncio.CancelledError:
            logger.info("Subscription cancelled.")
        except Exception as e:
            logger.error(f"Subscription error: {e}")
        finally:
            logger.info("Subscription stopped.")

    async def publish_job(self, job: RunRequest) -> str:
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

    def _process_message_callback(self, message: Message):
        """Callback to process a message received from Pub/Sub."""
        try:
            if not self.is_system_idle() or self._active_jobs >= self.max_concurrent_jobs:
                logger.debug(f"System busy or max jobs ({self._active_jobs}) reached. Nacking message {message.message_id}.")
                message.nack()
                return

            data = json.loads(message.data.decode("utf-8"))

            if data.get("type") != "job_request":
                logger.warning(f"Ignoring message {message.message_id} with type {data.get('type')}")
                message.ack()
                return

            self._active_jobs += 1
            logger.debug(f"Starting job processing. Active jobs: {self._active_jobs}")

            try:
                run_request = RunRequest.model_validate(data)
            except Exception as e:
                logger.error(f"Error creating RunRequest from message {message.message_id}: {e}")
                message.ack()
                self._active_jobs -= 1
                logger.debug(f"Job validation failed. Active jobs: {self._active_jobs}")
                return

            logger.info(f"Processing job {run_request.batch_id}:{run_request.record_id} from message {message.message_id}")

            message.ack()
            logger.debug(f"Acknowledged message {message.message_id}")

            loop = asyncio.get_running_loop()
            loop.create_task(self._run_job(run_request))
            self.publish_status_update(
                batch_id=run_request.batch_id,
                record_id=run_request.record_id,
                status=BatchJobStatus.RUNNING,
            )

        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}", exc_info=True)
            try:
                if self._active_jobs > 0 and "run_request" not in locals():
                    self._active_jobs -= 1
                message.nack()
                logger.warning(f"Nacked message {message.message_id} due to processing error.")
            except Exception as nack_e:
                logger.error(f"Failed to nack message {message.message_id} after error: {nack_e}")
        finally:
            if "run_request" in locals():
                self._active_jobs -= 1
                logger.debug(f"Finished job processing. Active jobs: {self._active_jobs}")

    def stop(self):
        """Stop the job worker daemon."""
        if not self._subscription_future:
            logger.warning("Job worker subscription not active or already stopped.")
            return

        logger.info("Stopping job worker subscription...")
        if self._subscription_future and not self._subscription_future.done():
            self._subscription_future.cancel()
        self._subscription_future = None
        logger.info("Job worker stopped.")

    def is_system_idle(self) -> bool:
        """Check if the system is idle and can process jobs.
        
        Checks both the ActivityTracker to see if there are any active
        user sessions, and the job queue's own idle status.
        
        Returns:
            True if the system is idle, False otherwise

        """
        activity_tracker = get_activity_tracker()
        return activity_tracker.is_idle()

    async def _run_job(self, run_request: RunRequest):
        """Run a job and update its status.

        Args:
            run_request: The RunRequest to execute

        """
        try:
            result = await self.flow_runner.run_flow(run_request)

            self.publish_status_update(
                batch_id=run_request.batch_id,
                record_id=run_request.record_id,
                status=BatchJobStatus.COMPLETED,
            )

            logger.info(f"Job {run_request.batch_id}:{run_request.record_id} completed successfully")

        except Exception as e:
            logger.error(f"Error running job {run_request.batch_id}:{run_request.record_id}: {e}", exc_info=True)

            self.publish_status_update(
                batch_id=run_request.batch_id,
                record_id=run_request.record_id,
                status=BatchJobStatus.FAILED,
                error=str(e),
            )
