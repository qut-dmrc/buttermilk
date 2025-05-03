"""Job queue client for interacting with Google Pub/Sub.

This module provides a client for publishing and subscribing to job messages
via Google Pub/Sub, allowing for decoupled job creation and execution.
"""

import json
from typing import Self

import pydantic
from google.cloud import pubsub
from pydantic import BaseModel, PrivateAttr

from buttermilk._core.batch import BatchJobDefinition, BatchJobStatus
from buttermilk.bm import bm, logger


class JobQueueClient(BaseModel):
    """Client for interacting with the Google Pub/Sub job queue.
    
    This class provides methods for:
    - Publishing job requests to the queue
    - Publishing job status updates
    - Checking if the system is idle
    """

    _publisher: pubsub.PublisherClient = PrivateAttr(default_factory=pubsub.PublisherClient)
    _subscriber: pubsub.SubscriberClient = PrivateAttr(default_factory=pubsub.SubscriberClient)
    _subscription_path: str = PrivateAttr()
    _status_topic_path: str = PrivateAttr()
    _jobs__topic_path: str = PrivateAttr()
    _is_processing: bool = PrivateAttr(default=False)

    @pydantic.model_validator(mode="after")
    def _subscribe(self) -> Self:
        self._subscription_path = self._subscriber.subscription_path(
            bm.pubsub.project,
            bm.pubsub.subscription,
        )
        self._status_topic_path = self._subscriber.topic_path(bm.pubsub.project, bm.pubsub.topic)
        self._jobs__topic_path = self._subscriber.topic_path(bm.pubsub.project, bm.pubsub.topic)

        # Track if we're currently processing a job
        self._is_processing = False

    def publish_job(self, job: BatchJobDefinition) -> str:
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
        future = self._publisher.publish(self._jobs__topic_path, message_data)
        message_id = future.result()

        logger.info(f"Published job {job.batch_id}:{job.record_id} to Pub/Sub (ID: {message_id})")
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

    def set_processing_status(self, is_processing: bool) -> None:
        """Set whether the system is currently processing a job.
        
        Args:
            is_processing: True if processing a job, False otherwise

        """
        self.is_processing = is_processing

    def is_idle(self) -> bool:
        """Check if the system is idle and available to process jobs.
        
        Returns:
            True if the system is idle, False otherwise

        """
        # Currently just checks if we're processing a job
        # This could be expanded to check other conditions like API activity
        return not self.is_processing
