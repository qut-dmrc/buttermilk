"""Tests for OpenAIBatchExecutor, client factory, and BatchLLMProcessor.

Tests cover:
- _create_openai_batch_client() with Azure, OpenAI, and xAI client types
- OpenAIBatchExecutor.execute() with mocked OpenAIBatchJobManager
- OpenAIBatchExecutor.get_status() with mocked manager
- Error cases (missing model, unsupported client_type, missing prepare_batch_requests)
- BatchLLMProcessor as provider-agnostic base for OpenAI executor
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk.batch.executors.openai import OpenAIBatchExecutor, _create_openai_batch_client
from buttermilk.batch.result import BatchJobStatus


# =============================================================================
# Fixtures
# =============================================================================


def _make_llm_config(client_type, api_key: str = "test-key", base_url: str | None = None, configs: dict | None = None):
    """Create a mock LLMConfig-like object."""
    return SimpleNamespace(
        client_type=client_type,
        api_key=api_key,
        base_url=base_url,
        configs=configs or {},
    )


def _make_mock_bm(connections: dict):
    """Create a mock bm object with llms.connections."""
    mock_bm = MagicMock()
    mock_bm.llms.connections = connections
    return mock_bm


# =============================================================================
# _create_openai_batch_client Tests
# =============================================================================


class TestCreateOpenAIBatchClient:
    """Tests for _create_openai_batch_client factory function."""

    def test_azure_client(self):
        """Azure client should return AzureOpenAI instance with /chat/completions endpoint."""
        from buttermilk._core.llms import ClientType

        mock_bm = _make_mock_bm({
            "gpt-chat": _make_llm_config(
                client_type=ClientType.AZURE,
                api_key="azure-key",
                base_url="https://my-resource.openai.azure.com/",
                configs={"api_version": "2024-12-01-preview"},
            ),
        })

        with (
            patch("buttermilk.bm", mock_bm),
            patch("openai.AzureOpenAI") as MockAzure,
        ):
            mock_client = MagicMock()
            MockAzure.return_value = mock_client

            client, endpoint = _create_openai_batch_client("gpt-chat")

            assert endpoint == "/chat/completions"
            MockAzure.assert_called_once_with(
                api_key="azure-key",
                azure_endpoint="https://my-resource.openai.azure.com/",
                api_version="2024-12-01-preview",
            )
            assert client is mock_client

    def test_openai_client(self):
        """OpenAI client should return OpenAI instance with /v1/chat/completions endpoint."""
        from buttermilk._core.llms import ClientType

        mock_bm = _make_mock_bm({
            "gpt-4o": _make_llm_config(
                client_type=ClientType.OPENAI,
                api_key="openai-key",
                base_url=None,
            ),
        })

        with (
            patch("buttermilk.bm", mock_bm),
            patch("openai.OpenAI") as MockOpenAI,
        ):
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client

            client, endpoint = _create_openai_batch_client("gpt-4o")

            assert endpoint == "/v1/chat/completions"
            MockOpenAI.assert_called_once_with(api_key="openai-key")
            assert client is mock_client

    def test_xai_client(self):
        """xAI client should return OpenAI instance with xAI base_url."""
        from buttermilk._core.llms import ClientType

        mock_bm = _make_mock_bm({
            "grok-fast": _make_llm_config(
                client_type=ClientType.XAI,
                api_key="xai-key",
                base_url="https://api.x.ai/v1",
            ),
        })

        with (
            patch("buttermilk.bm", mock_bm),
            patch("openai.OpenAI") as MockOpenAI,
        ):
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client

            client, endpoint = _create_openai_batch_client("grok-fast")

            assert endpoint == "/v1/chat/completions"
            MockOpenAI.assert_called_once_with(
                api_key="xai-key",
                base_url="https://api.x.ai/v1",
            )
            assert client is mock_client

    def test_missing_model_raises(self):
        """Missing model should raise ValueError."""
        mock_bm = _make_mock_bm({})

        with patch("buttermilk.bm", mock_bm):
            with pytest.raises(ValueError, match="not found in buttermilk connections"):
                _create_openai_batch_client("nonexistent-model")

    def test_unsupported_client_type_raises(self):
        """Unsupported client type should raise ValueError."""
        from buttermilk._core.llms import ClientType

        mock_bm = _make_mock_bm({
            "gemini-model": _make_llm_config(
                client_type=ClientType.GEMINI,
            ),
        })

        with patch("buttermilk.bm", mock_bm):
            with pytest.raises(ValueError, match="not supported for OpenAI batch"):
                _create_openai_batch_client("gemini-model")

    def test_azure_default_api_version(self):
        """Azure client without explicit api_version should use default."""
        from buttermilk._core.llms import ClientType

        mock_bm = _make_mock_bm({
            "gpt-mini": _make_llm_config(
                client_type=ClientType.AZURE,
                api_key="azure-key",
                base_url="https://my-resource.openai.azure.com/",
                configs={},  # No api_version
            ),
        })

        with (
            patch("buttermilk.bm", mock_bm),
            patch("openai.AzureOpenAI") as MockAzure,
        ):
            MockAzure.return_value = MagicMock()
            _create_openai_batch_client("gpt-mini")

            call_kwargs = MockAzure.call_args[1]
            assert call_kwargs["api_version"] == "2024-12-01-preview"


# =============================================================================
# OpenAIBatchExecutor Tests
# =============================================================================


class TestOpenAIBatchExecutor:
    """Tests for OpenAIBatchExecutor."""

    @pytest.fixture
    def executor(self):
        return OpenAIBatchExecutor(poll_interval=10, max_wait_hours=1)

    @pytest.mark.anyio
    async def test_execute_missing_prepare_batch_requests(self, executor):
        """Processor without prepare_batch_requests should fail."""
        processor = MagicMock()
        processor.name = "test_processor"
        del processor.prepare_batch_requests

        result = await executor.execute(records=[], processor=processor)

        assert result.status == BatchJobStatus.FAILED
        assert "prepare_batch_requests" in result.error

    @pytest.mark.anyio
    async def test_execute_missing_model(self, executor):
        """Processor without model attribute should fail."""
        processor = MagicMock()
        processor.name = "test_processor"
        processor.prepare_batch_requests.return_value = [MagicMock()]
        processor.model = None

        result = await executor.execute(records=[MagicMock()], processor=processor)

        assert result.status == BatchJobStatus.FAILED
        assert "missing 'model' attribute" in result.error

    @pytest.mark.anyio
    async def test_execute_success(self, executor):
        """Successful batch submission should return PENDING status."""
        processor = MagicMock()
        processor.name = "test_processor"
        processor.model = "gpt-chat"
        processor.prepare_batch_requests.return_value = [MagicMock(), MagicMock()]

        mock_manager = MagicMock()
        mock_manager.submit_batch = AsyncMock(return_value={
            "openai_batch_id": "batch_abc123",
            "job_id": "batch_20260219_xyz",
        })

        executor._get_manager = MagicMock(return_value=mock_manager)

        result = await executor.execute(records=[MagicMock(), MagicMock()], processor=processor)

        assert result.status == BatchJobStatus.PENDING
        assert result.job_id == "batch_20260219_xyz"
        assert result.metadata["openai_batch_id"] == "batch_abc123"
        assert result.metadata["request_count"] == 2
        assert result.metadata["model"] == "gpt-chat"

    @pytest.mark.anyio
    async def test_execute_submission_error(self, executor):
        """Submission error should return FAILED status."""
        processor = MagicMock()
        processor.name = "test_processor"
        processor.model = "gpt-chat"
        processor.prepare_batch_requests.return_value = [MagicMock()]

        mock_manager = MagicMock()
        mock_manager.submit_batch = AsyncMock(side_effect=RuntimeError("API error"))

        executor._get_manager = MagicMock(return_value=mock_manager)

        result = await executor.execute(records=[MagicMock()], processor=processor)

        assert result.status == BatchJobStatus.FAILED
        assert "API error" in result.error

    @pytest.mark.anyio
    async def test_get_status_completed(self, executor):
        """Completed job should return COMPLETED status."""
        mock_manifest = MagicMock()
        mock_manifest.openai_batch_id = "batch_abc123"

        mock_manager = MagicMock()
        mock_manager._load_manifest.return_value = mock_manifest
        mock_manager.get_batch_status.return_value = {"status": "completed"}

        executor._managers = {"gpt-chat": mock_manager}

        status = await executor.get_status("batch_20260219_xyz")
        assert status == BatchJobStatus.COMPLETED

    @pytest.mark.anyio
    async def test_get_status_in_progress(self, executor):
        """In-progress job should return RUNNING status."""
        mock_manifest = MagicMock()
        mock_manifest.openai_batch_id = "batch_abc123"

        mock_manager = MagicMock()
        mock_manager._load_manifest.return_value = mock_manifest
        mock_manager.get_batch_status.return_value = {"status": "in_progress"}

        executor._managers = {"gpt-chat": mock_manager}

        status = await executor.get_status("batch_20260219_xyz")
        assert status == BatchJobStatus.RUNNING

    @pytest.mark.anyio
    async def test_get_status_failed(self, executor):
        """Failed job should return FAILED status."""
        mock_manifest = MagicMock()
        mock_manifest.openai_batch_id = "batch_abc123"

        mock_manager = MagicMock()
        mock_manager._load_manifest.return_value = mock_manifest
        mock_manager.get_batch_status.return_value = {"status": "failed"}

        executor._managers = {"gpt-chat": mock_manager}

        status = await executor.get_status("batch_20260219_xyz")
        assert status == BatchJobStatus.FAILED

    @pytest.mark.anyio
    async def test_get_status_no_manager(self, executor):
        """Job with no matching manager should return FAILED."""
        executor._managers = {}

        status = await executor.get_status("nonexistent_job")
        assert status == BatchJobStatus.FAILED

    @pytest.mark.anyio
    async def test_get_status_manifest_not_found(self, executor):
        """Manager that can't find manifest should try next manager."""
        mock_manager = MagicMock()
        mock_manager._load_manifest.side_effect = FileNotFoundError("Not found")

        executor._managers = {"gpt-chat": mock_manager}

        status = await executor.get_status("missing_job")
        assert status == BatchJobStatus.FAILED

    def test_manager_caching(self, executor):
        """Manager should be cached per model name."""
        mock_client = MagicMock()
        mock_endpoint = "/v1/chat/completions"

        with patch("buttermilk.batch.executors.openai._create_openai_batch_client") as mock_factory:
            mock_factory.return_value = (mock_client, mock_endpoint)

            manager1 = executor._get_manager("gpt-chat")
            manager2 = executor._get_manager("gpt-chat")

            assert manager1 is manager2
            mock_factory.assert_called_once_with("gpt-chat")

    def test_manager_different_models(self, executor):
        """Different models should get different managers."""
        mock_client = MagicMock()

        with patch("buttermilk.batch.executors.openai._create_openai_batch_client") as mock_factory:
            mock_factory.return_value = (mock_client, "/v1/chat/completions")

            manager1 = executor._get_manager("gpt-chat")
            manager2 = executor._get_manager("gpt-mini")

            assert manager1 is not manager2
            assert mock_factory.call_count == 2


# =============================================================================
# __init__.py Imports Test
# =============================================================================


class TestExecutorPackageImports:
    """Tests for executor package __init__.py exports."""

    def test_import_batch_executor(self):
        from buttermilk.batch.executors import BatchExecutor
        assert BatchExecutor is not None

    def test_import_sync_executor(self):
        from buttermilk.batch.executors import SyncBatchExecutor
        assert SyncBatchExecutor is not None

    def test_import_vertex_executor(self):
        from buttermilk.batch.executors import VertexBatchExecutor
        assert VertexBatchExecutor is not None

    def test_import_openai_executor(self):
        from buttermilk.batch.executors import OpenAIBatchExecutor
        assert OpenAIBatchExecutor is not None


# =============================================================================
# BatchLLMProcessor Tests
# =============================================================================


class TestBatchLLMProcessor:
    """Tests for BatchLLMProcessor as provider-agnostic base class."""

    def test_import_from_processors_package(self):
        """BatchLLMProcessor should be importable from buttermilk.processors."""
        from buttermilk.processors import BatchLLMProcessor
        assert BatchLLMProcessor is not None

    def test_import_from_vertex_batch_module(self):
        """BatchLLMProcessor should be importable from the vertex_batch module."""
        from buttermilk.processors.vertex_batch import BatchLLMProcessor
        assert BatchLLMProcessor is not None

    def test_class_hierarchy(self):
        """VertexBatchProcessor should be a subclass of BatchLLMProcessor."""
        from buttermilk.processors.vertex_batch import BatchLLMProcessor, VertexBatchProcessor
        assert issubclass(VertexBatchProcessor, BatchLLMProcessor)

    def test_vertex_still_importable_from_processors(self):
        """VertexBatchProcessor should still be importable from buttermilk.processors."""
        from buttermilk.processors import VertexBatchProcessor
        assert VertexBatchProcessor is not None

    def test_batch_llm_processor_instantiation(self):
        """BatchLLMProcessor should instantiate with required fields."""
        from buttermilk.processors.vertex_batch import BatchLLMProcessor

        processor = BatchLLMProcessor(
            model="gpt-chat",
            template="test_template",
        )
        assert processor.model == "gpt-chat"
        assert processor.template == "test_template"

    def test_batch_llm_processor_has_no_dry_run(self):
        """BatchLLMProcessor should NOT have dry_run field (Vertex-specific)."""
        from buttermilk.processors.vertex_batch import BatchLLMProcessor

        assert "dry_run" not in BatchLLMProcessor.model_fields

    def test_vertex_processor_has_dry_run(self):
        """VertexBatchProcessor should have dry_run field."""
        from buttermilk.processors.vertex_batch import VertexBatchProcessor

        processor = VertexBatchProcessor(
            model="gemini-2.5-flash",
            template="test_template",
            dry_run=True,
        )
        assert processor.dry_run is True

    @pytest.mark.anyio
    async def test_batch_llm_processor_process_batch_raises(self):
        """BatchLLMProcessor._process_batch() should raise NotImplementedError."""
        from buttermilk.processors.vertex_batch import BatchLLMProcessor

        processor = BatchLLMProcessor(
            model="gpt-chat",
            template="test_template",
        )

        with pytest.raises(NotImplementedError, match="not implemented"):
            await processor._process_batch([])

    def test_batch_llm_processor_has_prepare_batch_requests(self):
        """BatchLLMProcessor should have prepare_batch_requests method."""
        from buttermilk.processors.vertex_batch import BatchLLMProcessor

        processor = BatchLLMProcessor(
            model="gpt-chat",
            template="test_template",
        )
        assert hasattr(processor, "prepare_batch_requests")
        assert callable(processor.prepare_batch_requests)

    @pytest.mark.anyio
    async def test_executor_accepts_batch_llm_processor(self):
        """OpenAIBatchExecutor should work with BatchLLMProcessor."""
        from buttermilk.processors.vertex_batch import BatchLLMProcessor

        processor = BatchLLMProcessor(
            model="gpt-chat",
            template="test_template",
        )

        executor = OpenAIBatchExecutor(poll_interval=10, max_wait_hours=1)

        mock_manager = MagicMock()
        mock_manager.submit_batch = AsyncMock(return_value={
            "openai_batch_id": "batch_abc123",
            "job_id": "batch_20260219_xyz",
        })
        executor._get_manager = MagicMock(return_value=mock_manager)

        # Use patch.object since Pydantic models don't allow direct attribute assignment
        with patch.object(
            type(processor),
            "prepare_batch_requests",
            return_value=[MagicMock(), MagicMock()],
        ) as mock_prepare:
            result = await executor.execute(records=[MagicMock(), MagicMock()], processor=processor)

            assert result.status == BatchJobStatus.PENDING
            assert result.job_id == "batch_20260219_xyz"
            mock_prepare.assert_called_once()
