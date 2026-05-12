"""Unit tests for OpenAIBatchProcessor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord
from buttermilk.processors.vertex_batch import OpenAIBatchProcessor


class SampleOutput(BaseModel):
    verdict: str
    reasoning: str


@pytest.fixture
def mock_bm():
    """Mock buttermilk global instance."""
    mock = MagicMock()
    mock.llms.connections = {}
    mock.session_info.save_dir = "gs://test-bucket/session"
    mock.session_info.save_dir_base = "gs://test-bucket"
    mock.session_info.project_name = "test-project"
    return mock


@pytest.fixture
def azure_config():
    """Mock Azure OpenAI connection config."""
    config = MagicMock()
    config.client_type.value = "azure"
    config.api_key = "test-key"
    config.base_url = "https://test.openai.azure.com/"
    config.configs = {
        "model": "gpt-4o-deployment",
        "api_version": "2024-12-01-preview",
        "region": "eastus",
    }
    return config


@pytest.fixture
def openai_config():
    """Mock direct OpenAI connection config."""
    config = MagicMock()
    config.client_type.value = "openai"
    config.api_key = "test-key"
    config.base_url = None
    config.configs = {
        "model": "gpt-4o",
    }
    return config


@pytest.fixture
def gemini_config():
    """Mock Gemini connection config (for rejection test)."""
    config = MagicMock()
    config.client_type.value = "vertex"
    config.configs = {"model": "gemini-1.5-flash"}
    return config


class TestOpenAIBatchProcessor:
    """Unit tests for OpenAIBatchProcessor."""

    def test_init_azure(self, mock_bm, azure_config):
        """Test instantiation with Azure OpenAI config."""
        mock_bm.llms.connections = {"gpt-azure": azure_config}

        with patch("buttermilk.bm", mock_bm):
            processor = OpenAIBatchProcessor(model="gpt-azure", template="test")
            # Trigger client initialization
            client = processor._ensure_openai_client()

            from openai import AzureOpenAI

            assert isinstance(client, AzureOpenAI)
            assert client.api_key == "test-key"
            assert str(client.base_url).startswith("https://test.openai.azure.com/")

    def test_init_openai(self, mock_bm, openai_config):
        """Test instantiation with direct OpenAI config."""
        mock_bm.llms.connections = {"gpt-direct": openai_config}

        with patch("buttermilk.bm", mock_bm):
            processor = OpenAIBatchProcessor(model="gpt-direct", template="test")
            client = processor._ensure_openai_client()

            from openai import OpenAI

            assert isinstance(client, OpenAI)
            assert client.api_key == "test-key"

    def test_rejection_non_openai(self, mock_bm, gemini_config):
        """Test rejection of non-OpenAI models."""
        mock_bm.llms.connections = {"gemini": gemini_config}

        with patch("buttermilk.bm", mock_bm):
            processor = OpenAIBatchProcessor(model="gemini", template="test")
            with pytest.raises(ValueError, match="OpenAIBatchProcessor requires an OpenAI or Azure model"):
                processor._ensure_openai_client()

    def test_get_batch_model_name(self, mock_bm, azure_config):
        """Test resolving the actual deployment name for Azure."""
        mock_bm.llms.connections = {"gpt-azure": azure_config}

        with patch("buttermilk.bm", mock_bm):
            processor = OpenAIBatchProcessor(model="gpt-azure", template="test")
            model_name = processor._get_batch_model_name()
            assert model_name == "gpt-4o-deployment"

    @pytest.mark.anyio
    async def test_process_batch_dry_run(self, mock_bm, openai_config):
        """Test dry_run mode returns correct metadata."""
        mock_bm.llms.connections = {"gpt": openai_config}

        # Mock the manager and its build_jsonl method
        mock_manager = MagicMock()
        mock_manager.build_jsonl.return_value = '{"test": "jsonl"}'
        mock_manager._resolve_batch_dir.return_value = "gs://test/batch"

        with (
            patch("buttermilk.bm", mock_bm),
            patch("buttermilk.processors.vertex_batch.OpenAIBatchProcessor._ensure_manager", return_value=mock_manager),
            patch("buttermilk.processors.vertex_batch.uuid") as mock_uuid,
            patch("buttermilk.utils.save.upload_text") as mock_upload,
        ):
            mock_uuid.uuid4().hex = "1234567890abcdef"
            mock_upload.return_value = "gs://test/batch/input.jsonl"

            processor = OpenAIBatchProcessor(model="gpt", template="test", dry_run=True)

            record = BaseRecord(record_id="rec1", content="hello", metadata={})
            context = ProcessingContext(record=record, session_id="test-session")

            # We need to mock render_template as well or provide a real template
            with patch("buttermilk.processors.vertex_batch.render_template") as mock_render:
                mock_render.return_value = MagicMock(rendered="System: hi\nUser: hello")

                results = await processor._process_batch([context])

            assert len(results) == 1
            assert results[0].metadata["dry_run"] is True
            assert results[0].metadata["batch_status"] == "dry_run"
            assert results[0].metadata["dry_run_uri"] == "gs://test/batch/input.jsonl"

    @pytest.mark.anyio
    async def test_process_batch_non_blocking(self, mock_bm, openai_config):
        """Test non-blocking mode returns pending records."""
        mock_bm.llms.connections = {"gpt": openai_config}

        mock_manager = MagicMock()
        mock_manager.submit_batch = AsyncMock(return_value={"openai_batch_id": "batch_123"})

        with (
            patch("buttermilk.bm", mock_bm),
            patch("buttermilk.processors.vertex_batch.OpenAIBatchProcessor._ensure_manager", return_value=mock_manager),
            patch("buttermilk.processors.vertex_batch.render_template") as mock_render,
        ):
            mock_render.return_value = MagicMock(rendered="System: hi\nUser: hello")

            processor = OpenAIBatchProcessor(model="gpt", template="test", wait_for_completion=False)

            record = BaseRecord(record_id="rec1", content="hello", metadata={})
            context = ProcessingContext(record=record, session_id="test-session")

            results = await processor._process_batch([context])

            assert len(results) == 1
            assert results[0].metadata["batch_job_id"] == "batch_123"
            assert results[0].metadata["batch_status"] == "pending"
            mock_manager.submit_batch.assert_called_once()

    @pytest.mark.anyio
    async def test_process_batch_blocking(self, mock_bm, openai_config):
        """Test blocking mode maps results back to records."""
        mock_bm.llms.connections = {"gpt": openai_config}

        from buttermilk._core.vertex_batch import BatchResult

        mock_manager = MagicMock()
        mock_results = [BatchResult(custom_id="id1", record_id="rec1", response="LLM says hello", usage={"total_tokens": 10}, cost_usd=0.001)]
        mock_manager.run_batch_and_wait = AsyncMock(return_value=mock_results)

        with (
            patch("buttermilk.bm", mock_bm),
            patch("buttermilk.processors.vertex_batch.OpenAIBatchProcessor._ensure_manager", return_value=mock_manager),
            patch("buttermilk.processors.vertex_batch.render_template") as mock_render,
            patch("buttermilk.processors.vertex_batch.uuid") as mock_uuid,
        ):
            mock_uuid.uuid4.return_value = MagicMock(hex="id1")
            mock_render.return_value = MagicMock(rendered="System: hi\nUser: hello")

            processor = OpenAIBatchProcessor(model="gpt", template="test", wait_for_completion=True)

            record = BaseRecord(record_id="rec1", content="hello", metadata={})
            context = ProcessingContext(record=record, session_id="test-session")

            results = await processor._process_batch([context])

            assert len(results) == 1
            assert results[0].metadata["llm_output"] == "LLM says hello"
            assert results[0].metadata["cost_usd"] == 0.001
            mock_manager.run_batch_and_wait.assert_called_once()
