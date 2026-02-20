import json
from unittest.mock import MagicMock, patch

import pytest

from buttermilk._core.dmrc import set_bm
from buttermilk._core.llms import ClientType
from buttermilk._core.vertex_batch import _DEEPSEEK_MODEL_PATTERNS, BatchJobManager, OpenAIMessageConverter, _is_deepseek_model
from buttermilk.batch.managers.openai import OpenAIBatchJobManager
from buttermilk.processors.openai_batch import OpenAIBatchProcessor


@pytest.fixture(scope="function", autouse=True)
def mock_bm():
    mock = MagicMock()
    mock.llms.connections = {}
    set_bm(mock)
    yield mock
    set_bm(None)  # Cleanup


def test_deepseek_model_patterns():
    assert "deepseek" in _DEEPSEEK_MODEL_PATTERNS
    assert "deepseek-ai" in _DEEPSEEK_MODEL_PATTERNS
    assert _is_deepseek_model("deepseek-r1")
    assert _is_deepseek_model("deepseek-ai/deepseek-r1")
    assert not _is_deepseek_model("gpt-4o")


def test_vertex_batch_manager_deepseek_path():
    # Mock BatchJobManager to test _get_vertex_model_path without full initialization
    manager = BatchJobManager(client=MagicMock())

    # Test path resolution logic
    path = manager._get_vertex_model_path("deepseek-r1")
    assert path == "publishers/deepseek-ai/models/deepseek-r1"

    path_with_prefix = manager._get_vertex_model_path("deepseek-ai/deepseek-r1")
    assert path_with_prefix == "publishers/deepseek-ai/models/deepseek-r1"


def test_openai_message_converter_endpoints():
    request = MagicMock()
    request.custom_id = "123"
    request.messages = []
    request.model = "gpt-4"
    request.response_schema = None

    # Test default endpoint (OpenAI/xAI)
    converter_default = OpenAIMessageConverter(endpoint="/v1/chat/completions")
    req_default = converter_default.build_request(request)
    assert req_default["url"] == "/v1/chat/completions"

    # Test Azure endpoint
    converter_azure = OpenAIMessageConverter(endpoint="/chat/completions")
    req_azure = converter_azure.build_request(request)
    assert req_azure["url"] == "/chat/completions"


def test_openai_batch_manager_jsonl_generation():
    # Test that manager passes endpoint to converter and generates correct JSONL
    # Updated: Must provide jsonl_url explicitly if it differs from default
    manager = OpenAIBatchJobManager(client=MagicMock(), endpoint="/chat/completions", jsonl_url="/chat/completions")

    request = MagicMock()
    request.custom_id = "123"
    request.messages = []
    request.model = "gpt-4"
    request.response_schema = None

    # Mock requests list
    requests = [request]

    jsonl_content = manager.build_jsonl(requests, "gpt-4")

    # Parse the JSONL line
    entry = json.loads(jsonl_content)
    assert entry["url"] == "/chat/completions"


@patch("openai.AzureOpenAI")
def test_openai_batch_processor_azure_init(mock_azure, mock_bm):
    # Setup mock LLM config for Azure
    mock_config = MagicMock()
    mock_config.client_type = ClientType.AZURE
    mock_config.api_key = "fake-key"
    mock_config.base_url = "https://my-azure.openai.azure.com/"
    mock_config.configs = {"api_version": "2024-02-15-preview"}

    mock_bm.llms.connections = {"azure-gpt4": mock_config}

    processor = OpenAIBatchProcessor(model="azure-gpt4", template="test_template")

    # Trigger manager initialization
    manager = processor._ensure_manager()

    # Verify Azure client was created
    mock_azure.assert_called_once_with(api_key="fake-key", azure_endpoint="https://my-azure.openai.azure.com/", api_version="2024-02-15-preview")

    # Verify endpoint is correct for Azure Batch
    assert manager.endpoint == "/chat/completions"


@patch("openai.OpenAI")
def test_openai_batch_processor_openai_init(mock_openai, mock_bm):
    # Setup mock LLM config for OpenAI
    mock_config = MagicMock()
    mock_config.client_type = ClientType.OPENAI
    mock_config.api_key = "sk-fake"
    mock_config.base_url = None

    mock_bm.llms.connections = {"gpt-4o": mock_config}

    processor = OpenAIBatchProcessor(model="gpt-4o", template="test_template")

    # Trigger manager initialization
    manager = processor._ensure_manager()

    # Verify OpenAI client was created
    mock_openai.assert_called_once_with(api_key="sk-fake", base_url=None)

    # Verify endpoint is correct for OpenAI Batch
    assert manager.endpoint == "/v1/chat/completions"


@patch("openai.OpenAI")
def test_openai_batch_processor_xai_init(mock_openai, mock_bm):
    # Setup mock LLM config for xAI (Grok)
    mock_config = MagicMock()
    mock_config.client_type = ClientType.XAI
    mock_config.api_key = "xai-fake"
    mock_config.base_url = "https://api.x.ai/v1"

    mock_bm.llms.connections = {"grok-2": mock_config}

    processor = OpenAIBatchProcessor(model="grok-2", template="test_template")

    # Trigger manager initialization
    manager = processor._ensure_manager()

    # Verify OpenAI client was created with xAI base_url
    mock_openai.assert_called_once_with(api_key="xai-fake", base_url="https://api.x.ai/v1")

    # Verify endpoint is correct for OpenAI-compatible Batch
    assert manager.endpoint == "/v1/chat/completions"
