"""OpenAI/Azure batch executor for BatchPipelineRunner.

Wraps OpenAIBatchJobManager to submit batch jobs via the OpenAI Batch API.
Works with both direct OpenAI and Azure OpenAI endpoints.
"""

from __future__ import annotations

from typing import Any

from buttermilk import logger
from buttermilk._core.processor_core import BatchProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.batch.executors.base import BatchExecutor
from buttermilk.batch.managers.openai import OpenAIBatchJobManager
from buttermilk.batch.result import BatchExecutionResult, BatchJobStatus


def _create_openai_batch_client(model_name: str) -> tuple[Any, str]:
    """Create an OpenAI/Azure client from the buttermilk model registry.

    Looks up the model in bm.llms.connections and constructs the appropriate
    OpenAI or AzureOpenAI client for the Batch API.

    Args:
        model_name: Model name as registered in buttermilk (e.g., "gpt-mini", "gpt-chat")

    Returns:
        Tuple of (client, endpoint) where endpoint is the batch API path.

    Raises:
        ValueError: If model not found or not an OpenAI/Azure model.
    """
    from buttermilk import bm
    from buttermilk._core.llms import ClientType

    if model_name not in bm.llms.connections:
        raise ValueError(f"Model '{model_name}' not found in buttermilk connections. Available: {list(bm.llms.connections.keys())}")

    config = bm.llms.connections[model_name]

    if config.client_type == ClientType.AZURE:
        from openai import AzureOpenAI

        if not config.base_url:
            raise ValueError(
                f"Model '{model_name}' is configured as Azure (ClientType.AZURE) but "
                f"'base_url' is not set. Please set 'base_url' to your Azure OpenAI endpoint URL "
                f"(e.g. 'https://<resource>.openai.azure.com/')."
            )
        if not config.api_key:
            raise ValueError(
                f"Model '{model_name}' is configured as Azure (ClientType.AZURE) but 'api_key' is not set. Please provide your Azure OpenAI API key."
            )
        api_version = config.configs.get("api_version", "2024-12-01-preview")
        client = AzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.base_url,
            api_version=api_version,
        )
        # Azure uses /chat/completions (no /v1/ prefix)
        return client, "/chat/completions"

    elif config.client_type == ClientType.OPENAI:
        from openai import OpenAI

        kwargs: dict[str, Any] = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        client = OpenAI(**kwargs)
        return client, "/v1/chat/completions"

    elif config.client_type == ClientType.XAI:
        from openai import OpenAI

        client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or "https://api.x.ai/v1",
        )
        return client, "/v1/chat/completions"

    else:
        raise ValueError(
            f"Model '{model_name}' has client_type '{config.client_type.value}', "
            f"which is not supported for OpenAI batch. "
            f"Expected 'openai', 'azure', or 'xai'."
        )


class OpenAIBatchExecutor(BatchExecutor):
    """Executes batch processing via the OpenAI/Azure Batch API.

    Submits a batch job and returns immediately with PENDING status.
    Requires a processor that supports `prepare_batch_requests`
    (e.g. BatchLLMProcessor or VertexBatchProcessor).

    The client is created lazily from the buttermilk model registry
    based on the processor's model name.
    """

    def __init__(self, poll_interval: int = 30, max_wait_hours: int = 24) -> None:
        self.poll_interval = poll_interval
        self.max_wait_hours = max_wait_hours
        self._managers: dict[str, OpenAIBatchJobManager] = {}

    def _get_manager(self, model_name: str) -> OpenAIBatchJobManager:
        """Get or create a manager for the given model.

        Creates the OpenAI/Azure client from buttermilk's model registry
        and caches it per model name.
        """
        if model_name not in self._managers:
            client, endpoint = _create_openai_batch_client(model_name)
            self._managers[model_name] = OpenAIBatchJobManager(
                client=client,
                endpoint=endpoint,
                poll_interval=self.poll_interval,
                max_wait_hours=self.max_wait_hours,
            )
        return self._managers[model_name]

    async def execute(
        self,
        records: list[BaseRecord],
        processor: BatchProcessorCore,
    ) -> BatchExecutionResult:
        """Submit batch job to OpenAI/Azure Batch API."""

        # 1. Validate Processor Compatibility
        if not hasattr(processor, "prepare_batch_requests"):
            return BatchExecutionResult(
                status=BatchJobStatus.FAILED,
                error=(
                    f"Processor {processor.name} ({type(processor).__name__}) does not support batch execution. Missing 'prepare_batch_requests'."
                ),
            )

        # 2. Prepare Requests
        try:
            requests = processor.prepare_batch_requests(records)  # type: ignore[attr-defined]

            model = getattr(processor, "model", None)
            if not model:
                return BatchExecutionResult(
                    status=BatchJobStatus.FAILED,
                    error=f"Processor {processor.name} missing 'model' attribute.",
                )

            # 3. Get manager (creates client from registry)
            manager = self._get_manager(model)

            # 4. Submit Job
            result = await manager.submit_batch(
                model=model,
                requests=requests,
            )

            openai_batch_id = result["openai_batch_id"]
            job_id = result["job_id"]
            logger.info(f"OpenAI batch job submitted: {openai_batch_id} (internal: {job_id})")

            return BatchExecutionResult(
                status=BatchJobStatus.PENDING,
                job_id=job_id,
                metadata={
                    "model": model,
                    "request_count": len(requests),
                    "openai_batch_id": openai_batch_id,
                },
            )

        except Exception as e:
            logger.error(f"Failed to submit OpenAI batch job: {e}")
            return BatchExecutionResult(
                status=BatchJobStatus.FAILED,
                error=str(e),
            )

    async def get_status(self, job_id: str) -> BatchJobStatus:
        """Get status of an OpenAI batch job.

        Loads manifest to find model and openai_batch_id, then checks status via API.
        This is process-restart safe as it doesn't depend on existing managers.
        """
        try:
            # 1. Load manifest to find model and openai_batch_id
            # We need a manager instance to call _load_manifest, but loading
            # doesn't actually use the client. Use any existing manager or
            # a temporary one.
            manager = None
            if self._managers:
                manager = next(iter(self._managers.values()))
            else:
                # Need to load manifest to find model, but OpenAIBatchJobManager
                # requires a client in init. We'll use a mock-like client
                # just for manifest loading.
                from buttermilk.batch.managers.openai import OpenAIBatchJobManager

                manager = OpenAIBatchJobManager(client=object(), endpoint="")

            manifest = manager._load_manifest(job_id)

            # 2. Get the real manager for the specific model
            real_manager = self._get_manager(manifest.model)

            # 3. Check status via API
            status_info = real_manager.get_batch_status(manifest.openai_batch_id)

            status_map = {
                "completed": BatchJobStatus.COMPLETED,
                "failed": BatchJobStatus.FAILED,
                "expired": BatchJobStatus.FAILED,
                "cancelled": BatchJobStatus.CANCELLED,
                "in_progress": BatchJobStatus.RUNNING,
                "validating": BatchJobStatus.PENDING,
                "finalizing": BatchJobStatus.RUNNING,
            }
            return status_map.get(status_info["status"], BatchJobStatus.RUNNING)

        except FileNotFoundError:
            logger.error(f"Manifest not found for job {job_id}")
            return BatchJobStatus.FAILED
        except Exception as e:
            logger.warning(f"Error checking job status for {job_id}: {e}")
            return BatchJobStatus.FAILED
