import asyncio
import importlib
import logging
import random
from collections.abc import AsyncGenerator, Callable
from typing import Any

import shortuuid
from fastapi import WebSocketDisconnect
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, ConfigDict, Field
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from buttermilk import logger
from buttermilk._core import (
    AgentTrace,
    logger,  # noqa
)
from buttermilk._core.agent import ErrorEvent
from buttermilk._core.config import SaveInfo
from buttermilk._core.context import set_logging_context
from buttermilk._core.contract import (
    ErrorEvent,
    FlowEvent,
    FlowMessage,
    UIMessage,
)
from buttermilk._core.exceptions import FatalError
from buttermilk._core.log import logger
from buttermilk._core.orchestrator import Orchestrator, OrchestratorProtocol
from buttermilk._core.types import Record, RunRequest
from buttermilk.api.job_queue import JobQueueClient
from buttermilk.api.services.data_service import DataService
from buttermilk.api.services.message_service import MessageService
from buttermilk.utils.utils import expand_dict


class FlowRunContext(BaseModel):
    """Encapsulates all state for a single flow run."""

    flow_name: str = ""
    flow_task: Any | None = None
    orchestrator: Orchestrator | None = None
    status: str = "pending"
    session_id: str
    callback_to_groupchat: Any = None
    messages: list = []
    progress: dict = Field(default_factory=dict)

    websocket: Any = None

    async def monitor_ui(self) -> AsyncGenerator[RunRequest, None]:
        """Monitor the UI for incoming messages."""
        while True:
            await asyncio.sleep(0.1)

            # Check if the WebSocket is connected
            if not self.websocket or self.websocket.client_state != WebSocketState.CONNECTED:
                logger.debug(f"WebSocket not connected for session {self.session_id}")
                continue

            try:
                data = await self.websocket.receive_json()
                message = await MessageService.process_message_from_ui(data)
                if not message:
                    continue

                if isinstance(message, RunRequest):
                    # Generate a request to run the flow with the new parameters
                    message.callback_to_ui = self.send_message_to_ui
                    message.session_id = self.session_id

                    yield message
                elif not self.callback_to_groupchat:
                    # Group chat has not started yet
                    logger.debug(f"Group chat not yet started for session {self.session_id}")
                    continue
                else:
                    await self.callback_to_groupchat(message)

            except WebSocketDisconnect:
                logger.info(f"Client {self.session_id} disconnected.")
                self.websocket = None
                break
            except Exception as e:
                logger.error(f"Error receiving/processing client message for {self.session_id}: {e}")
                self.websocket = None
                break
                # raise FatalError(f"Error receiving/processing client message for {self.session_id}: {e}")

    async def send_message_to_ui(self, message: AgentTrace | UIMessage | Record | FlowEvent | FlowMessage) -> None:
        """Send a message to a WebSocket connection.

        Args:
            message: The message to send

        """
        formatted_message = MessageService.format_message_for_client(message)
        if not formatted_message:
            logger.debug(f"Dropping message not handled by client: {message}")
            return

        try:
            message_type = formatted_message.type
            message_data_to_send = formatted_message.model_dump(mode="json", exclude_unset=True, exclude_none=True)

            @retry(
                stop=stop_after_attempt(30),  # Try 30 times in total (1 initial + 29 retries)
                wait=wait_fixed(2),          # Wait 2 seconds between attempts
                retry=retry_if_exception_type(Exception),  # Retry on any exception during send
                reraise=True,                # Reraise the last exception if all retries fail
                before_sleep=before_sleep_log(logger, logging.WARNING),  # Log a warning before retrying
            )
            async def _send_with_retry_internal():
                if not self.websocket:
                    logger.error(f"WebSocket is None for session {self.session_id} during send attempt.")
                    # Raise an error to be caught by tenacity or the outer try/except
                    raise RuntimeError(f"WebSocket is None for session {self.session_id}")

                if self.websocket.client_state != WebSocketState.CONNECTED:
                    logger.warning(
                        f"WebSocket not connected (state: {self.websocket.client_state}) for session {self.session_id}. Retrying send.",
                    )
                    # Proceed to send; if it fails due to state, tenacity will catch and retry.

                await self.websocket.send_json(message_data_to_send)
                logger.debug(f"Message sent to UI for session {self.session_id}: {message_type}")

            if not self.websocket:
                logger.error(f"Cannot send message to UI for session {self.session_id}: WebSocket is None.")
                # No point in trying to send an error message if websocket is None
                return

            await _send_with_retry_internal()

        except Exception as e:
            logger.error(f"Error sending message to UI for session {self.session_id} (final attempt or pre-send issue): {e}")
            # Attempt to send an error message back to the client if the websocket is still viable
            if self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
                try:
                    error_event = ErrorEvent(source="websocket_manager", content=f"Failed to send message to client: {e!s}")
                    error_message_data = {"content": error_event.model_dump(), "type": "system_message"}
                    await self.websocket.send_json(error_message_data)
                except Exception as e_fallback:
                    logger.error(f"Failed to send error notification to UI for session {self.session_id}: {e_fallback!s}")
            else:
                logger.warning(f"Cannot send error notification to UI for session {self.session_id}: WebSocket not available or not connected.")


class FlowRunner(BaseModel):
    """Centralized service for running flows across different entry points.

    Handles orchestrator instantiation and execution in a consistent way, regardless
    of whether the flow is started from CLI, API, Slackbot, or Pub/Sub.
    """

    flows: dict[str, OrchestratorProtocol] = Field(default_factory=dict)  # Flow configurations

    save: SaveInfo
    tasks: list = Field(default=[])
    model_config = ConfigDict(extra="allow")
    mode: str = Field(default="api")
    ui: str = Field(default="console")
    human_in_loop: bool = False
    sessions: dict[str, FlowRunContext] = Field(default_factory=dict)  # Dictionary of active sessions

    def get_websocket_session(self, session_id: str, websocket: Any | None = None) -> FlowRunContext | None:
        """Get or create a session for the given session ID.

        Args:
            session_id: Unique identifier for the session
            websocket: WebSocket connection for this session

        Returns:
            A FlowRunContext object representing the session

        """
        if session_id not in self.sessions and not websocket:
            return None
        if session_id not in self.sessions:
            logger.info(f"Creating new session for {session_id} with WebSocket")
            self.sessions[session_id] = FlowRunContext(session_id=session_id, websocket=websocket)
        elif websocket is not None:
            # Replace the existing WebSocket connection
            logger.info(f"Updating WebSocket for existing session {session_id} ({websocket.client_state})")
            self.sessions[session_id].websocket = websocket
        return self.sessions[session_id]

    async def pull_and_run_task(self) -> None:
        """Pull tasks from the queue and run them."""
        # Initialize the queue_manager if needed
        queue_manager = getattr(self, "queue_manager", None)
        if queue_manager is None:
            self.queue_manager = JobQueueClient()

        # Pull task from the queue
        request = await self.queue_manager.pull_single_task()

        raise FatalError("Need to create the sssion object first")
        if request:
            # Run the task with a fresh orchestrator
            logger.info(f"Running task from queue: {request.flow} (Task ID: {request.job_id})")  # Updated message
            await self.run_flow(request, wait_for_completion=True)
        else:
            logger.debug("No tasks available in the queue")

    def _create_fresh_orchestrator(self, flow_name: str) -> OrchestratorProtocol:
        """Create a completely fresh orchestrator instance.

        Args:
            flow_name: The name of the flow to create an orchestrator for
            session_id: The session ID for the flow

        Returns:
            A new orchestrator instance with fresh state

        Raises:
            ValueError: If flow_name doesn't exist in flows

        """
        if flow_name not in self.flows:
            raise ValueError(f"Flow '{flow_name}' not found. Available flows: {list(self.flows.keys())}")

        flow_config = self.flows[flow_name]

        # Extract orchestrator class path
        orchestrator_path = flow_config.orchestrator
        module_name, class_name = orchestrator_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        orchestrator_cls = getattr(module, class_name)

        # Create a fresh config copy to avoid shared state
        config = flow_config.model_dump() if hasattr(flow_config, "model_dump") else dict(flow_config)

        # Create and return a fresh instance
        return orchestrator_cls(**config)

    async def _cleanup_flow_context(self, context: FlowRunContext) -> None:
        """Clean up resources associated with a flow run.

        Args:
            context: The flow run context to clean up

        """
        # Implement cleanup logic for the orchestrator if available
        cleanup_method = getattr(context.orchestrator, "cleanup", None)
        if cleanup_method is not None and callable(cleanup_method):
            try:
                result = cleanup_method()
                # Handle case where cleanup might be async or not
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Error during orchestrator cleanup: {e}")

        # Set status to completed
        context.status = "completed"

    async def run_flow(self,
                      run_request: RunRequest,
                      wait_for_completion: bool = False,
                      **kwargs) -> None:
        """Run a flow based on its configuration and a request.

        Args:
            run_request: The request containing input parameters
            wait_for_completion: If True, await the flow's completion before returning.
                                 If False (default), start the flow as a background
                                 task and return immediately.
            history: Optional conversation history (for chat-based interfaces)
            **kwargs: Additional keyword arguments for orchestrator instantiation

        Returns:
            If wait_for_completion is True, returns the result of the orchestrator run.
            If wait_for_completion is False, returns a callback function.

        Raises:
            ValueError: If orchestrator isn't specified or unknown

        """
        # Create a fresh orchestrator instance
        fresh_orchestrator = self._create_fresh_orchestrator(run_request.flow)

        # set a high max callback duration when dealing with LLMs
        asyncio.get_event_loop().slow_callback_duration = 120

        # Type safety: The orchestrator will be an Orchestrator instance at runtime,
        # even though the flows dict is typed with the more general OrchestratorProtocol
        if run_request.session_id not in self.sessions:
            # Create a new session if it doesn't exist
            self.sessions[run_request.session_id] = FlowRunContext(session_id=run_request.session_id)
        set_logging_context(run_request.session_id)
        _session = self.sessions[run_request.session_id]
        _session.flow_name = run_request.flow
        _session.orchestrator = fresh_orchestrator
        _session.callback_to_groupchat = fresh_orchestrator.make_publish_callback()

        # Create the task
        _session.flow_task = asyncio.create_task(fresh_orchestrator.run(request=run_request))  # type: ignore

        # ======== MAJOR EVENT: FLOW STARTING ========
        # Log detailed information about flow start
        logger.info(f"🚀 FLOW STARTING: '{run_request.flow}' (ID: {run_request.job_id}).\n📋 RunRequest: {run_request.model_dump_json(indent=2)}\n⚙️ Source: {', '.join(run_request.source) if run_request.source else 'direct'}\n✅ New flow instance created - all state has been reset")

        try:
            if wait_for_completion:
                # Wait for the task
                await _session.flow_task
                _session.status = "completed"
                return
        except Exception as e:
            _session.status = "failed"
            logger.error(f"Error running flow '{run_request.flow}': {e}")
            raise
        finally:
            if wait_for_completion:
                # Clean up after completion if we were waiting
                await self._cleanup_flow_context(_session)
        return

    async def create_batch(self, flow_name, max_records: int | None) -> list[RunRequest]:
        """Create a new batch job from the given request.

        Args:
            batch_request: The batch configuration

        Returns:
            The created batch metadata

        Raises:
            ValueError: If the flow doesn't exist or record extraction fails

        """
        # Extract record IDs from the flow's data source
        record_ids = await DataService.get_records_for_flow(flow_name=flow_name, flow_runner=self)
        logger.info(f"Extracted {len(record_ids)} record IDs for flow '{flow_name}'")

        # Create multiple iterations by multiplying the parameters
        iteration_values = expand_dict(self.flows[flow_name].parameters) or [{}]
        logger.debug(f"Expanded {len(self.flows[flow_name].parameters)} parameters for batch into {len(iteration_values)} variants")

        # Shuffle records
        random.shuffle(record_ids)
        logger.debug(f"Shuffled {len(record_ids)} record IDs")

        batch_id = str(shortuuid.uuid())

        # Create run requests for each record and parameter combination
        job_definitions = []

        # Apply iteration values
        for iteration_params in iteration_values:
            for i, record in enumerate(record_ids):
                job = RunRequest(ui_type="batch",
                    batch_id=batch_id,
                    flow=flow_name,
                    record_id=record["record_id"],
                    parameters=iteration_params, callback_to_ui=None,
                )
                job_definitions.append(job)
                logger.debug(f"Created run request: {job.model_dump_json()}")

                # Apply max_records limit if specified
                if max_records is not None and max_records > 0 and i >= max_records:
                    break

        if max_records is not None and max_records > 0:
            logger.info(f"Limited to {max_records} record IDs, returning {len(iteration_values)} iterations for {len(job_definitions)} jobs total.")
        else:
            logger.info(f"Returning {len(iteration_values)} iterations for {len(job_definitions)} jobs total.")

        random.shuffle(job_definitions)

        try:
            # Enqueue the batch for processing
            job_queue = JobQueueClient()

            for request in job_definitions:
                job_queue.publish_job(request)

        except Exception as e:
            msg = f"Failed to publish job to queue: {e}"
            raise FatalError(msg) from e

        return job_definitions

    async def run_batch_job(self, callback_to_ui: Callable, max_jobs: int = 1, wait_for_completion: bool = True) -> None:
        """Pull and run jobs from the queue, ensuring fresh state for each job.

        Args:
            max_jobs: Maximum number of jobs to process in this batch run

        Raises:
            FatalError: If no run requests are found in the queue
            Exception: If there's an error running a job

        """
        try:
            worker = JobQueueClient(
                max_concurrent_jobs=1,  # Process one job at a time to maintain isolation
            )

            jobs_processed = 0

            while jobs_processed < max_jobs:
                # Pull a job from the queue
                run_request = await worker.pull_single_task()
                if not run_request:
                    if jobs_processed == 0:
                        # Only raise an error if we didn't process any jobs
                        raise FatalError("No run request found in the queue.")
                    break  # No more jobs to process

                run_request.callback_to_ui = callback_to_ui

                logger.info(f"Processing batch job {jobs_processed + 1}/{max_jobs}: {run_request.flow} (Job ID: {run_request.job_id})")
                try:
                    await self.run_flow(run_request=run_request, wait_for_completion=wait_for_completion)
                    if wait_for_completion:
                        logger.info(f"Successfully completed job {run_request.job_id}")
                    else:
                        logger.info(f"Job {run_request.job_id} started in the background")
                except Exception as job_error:
                    logger.error(f"Error running job {run_request.job_id}: {job_error}")
                    # Continue processing other jobs even if one fails

                jobs_processed += 1

            logger.info(f"Batch processing complete. Processed {jobs_processed} jobs.")

        except FatalError:
            # Re-raise FatalError to be handled by the caller
            raise
        except Exception as e:
            logger.error(f"Fatal error during batch processing: {e}")
            raise
