"""Task queue client for interacting with Google Pub/Sub.

This module provides a client for publishing and subscribing to job messages
via Google Pub/Sub, allowing for decoupled job creation and execution. It
also contains the daemon that monitors the Pub/Sub queue for jobs
and processes them when the system is idle (no interactive API or websocket activity).
"""

import asyncio
import json
import uuid
from typing import Any, Self

import pydantic
from google.cloud.pubsub import PublisherClient, SubscriberClient
from pydantic import BaseModel, PrivateAttr

from buttermilk._core import logger
from buttermilk._core.batch import BatchJobStatus
from buttermilk._core.types import RunRequest
from buttermilk.toxicity.tox_data import toxic_record


class JobQueueClient(BaseModel):
    """Client for interacting with the Google Pub/Sub job queue.

    This class provides methods for:
    - Publishing job requests to the queue
    - Publishing job status updates
    - Checking if the system is idle
    """

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
        from buttermilk import buttermilk
        self._jobs_subscription_path = self._subscriber.subscription_path(
            buttermilk.pubsub.project,
            buttermilk.pubsub.jobs_subscription,
        )
        self._status_subscription_path = self._subscriber.subscription_path(
            buttermilk.pubsub.project,
            buttermilk.pubsub.status_subscription,
        )
        self._status_topic_path = self._subscriber.topic_path(buttermilk.pubsub.project, buttermilk.pubsub.status_topic)
        self._jobs_topic_path = self._publisher.topic_path(buttermilk.pubsub.project, buttermilk.pubsub.jobs_topic)

        return self

    async def pull_tox_example(self) -> RunRequest:
        record = toxic_record()

        data = {"ui_type": "web", "parameters": {"criteria": "criteria_ordinary"}}
        request = RunRequest(flow="tox", records=[record], **data)
        return request

    async def pull_single_task(self) -> RunRequest | None:
        """Pull a single task from Pub/Sub and process it."""
        response = None
        try:
            response = self._subscriber.pull(subscription=self._jobs_subscription_path, max_messages=1)
            ack_id = response.received_messages[0].ack_id
            self._subscriber.acknowledge(subscription=self._jobs_subscription_path, ack_ids=[ack_id])
            logger.debug(f"Acknowledged message with ack_id: {ack_id}")

            if response.received_messages:
                message = response.received_messages[0]
                request = await self._make_run_request(message)
                return request
        except Exception as e:
            logger.error(f"Error pulling pub/sub message: {e}", exc_info=True)
        return None

    async def fetch_and_run_task(self) -> None:
        if not self.is_system_idle() or self._active_jobs >= self.max_concurrent_jobs:
            logger.info(f"System busy (Idle: {self.is_system_idle()}, Active Jobs: {self._active_jobs}/{self.max_concurrent_jobs}). Not processing job now.")
            self._processing.clear()
            return

        request = await self.pull_single_task()
        if request:
            await self._process_run_request(run_request=request, wait=True)

    async def pull_tasks(self) -> None:
        """Continuously pull tasks from Pub/Sub when the system is idle."""
        logger.info(f"Started job worker processing. Listening for messages on {self._jobs_subscription_path}...")
        while True:
            if self.is_system_idle() and not self._processing.is_set():
                try:
                    await self.fetch_and_run_task()
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
        """Publish a task status update.

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

        logger.debug(f"Published status update for task {batch_id}:{record_id}: {status}")
        return message_id

    async def _make_run_request(self, message: Any) -> RunRequest | None:
        ack_id = message.ack_id
        message_data = message.message.data

        try:
            # Add default fields that are likely missing because we're in batch mode
            data = {"ui_type": "web", "session_id": uuid.uuid4().hex, "callback_to_ui": None}

            # Decode the message data and parse it into a RunRequest
            incoming = json.loads(message_data.decode("utf-8"))
            incoming = {k: v for k, v in incoming.items() if v}
            data.update(incoming)
            run_request = RunRequest.model_validate(data)
            return run_request
        except pydantic.ValidationError as e:
            logger.error(f"Validation error creating RunRequest from message (ack_id: {ack_id}): {e}. Message data: {message_data}")
            try:
                self._subscriber.acknowledge(subscription=self._jobs_subscription_path, ack_ids=[ack_id])
                logger.warning(f"Acked malformed message (ack_id: {ack_id})")
            except Exception as ack_err:
                logger.error(f"Failed to ack malformed message (ack_id: {ack_id}): {ack_err}")
        return None

    async def _process_run_request(self, run_request: RunRequest, wait: bool = False) -> None:
        """Parse and dispatch a single message from the Pub/Sub queue."""
        self._active_jobs += 1
        try:
            logger.debug(f"Incremented active tasks to {self._active_jobs} for task processing.")
            logger.info(f"Processing task {run_request.batch_id or 'N/A'}:{run_request.record_id or 'N/A'} (Task ID: {run_request.job_id})")
            if not wait:
                loop = asyncio.get_running_loop()
                loop.create_task(self._run_job(run_request))
            else:
                await self._run_job(run_request)

            self.publish_status_update(
                batch_id=run_request.batch_id or "N/A",
                record_id=run_request.record_id or "N/A",
                status=BatchJobStatus.RUNNING,
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            if self._active_jobs > 0:
                self._active_jobs -= 1
                logger.warning(f"Decremented active tasks to {self._active_jobs} due to top-level processing error.")
            self._processing.clear()
            logger.warning("Cleared processing flag due to top-level error.")

    def is_system_idle(self) -> bool:
        """Check if the system is idle (no active user sessions)."""
        from buttermilk.web.activity_tracker import get_instance as get_activity_tracker
        activity_tracker = get_activity_tracker()
        return activity_tracker.is_idle()

    async def _run_job(self, run_request: RunRequest) -> None:
        """Run a task, update its status, and acknowledge the message upon success."""
        job_desc = f"{run_request.batch_id or 'N/A'}:{run_request.record_id or 'N/A'} (Task ID: {run_request.job_id})"

        try:
            logger.debug(f"Starting flow execution for task {job_desc}")

            # NOT IMPLEMENTED

            logger.debug(f"Flow execution finished for task {job_desc}. Result type: {type(result)}")

            self.publish_status_update(
                batch_id=run_request.batch_id or "N/A",
                record_id=run_request.record_id or "N/A",
                status=BatchJobStatus.COMPLETED,
            )

            logger.info(f"Task {job_desc} completed successfully")

        except Exception as e:
            logger.error(f"Error running task {job_desc}: {e}", exc_info=True)

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
                logger.warning(f"Attempted to decrement active_tasks below zero for task {job_desc}. State might be inconsistent.")

            if self._active_jobs == 0:
                self._processing.clear()
                logger.debug("Cleared processing flag as active jobs reached 0.")
            else:
                logger.debug(f"Processing flag remains set. Active tasks: {self._active_jobs}")

            logger.debug(f"Finished _run_job for {job_desc}. Active tasks: {self._active_jobs}")
