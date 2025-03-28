from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from buttermilk._core.contract import AgentInput, AgentOutput, UserInstructions
from buttermilk._core.runner_types import Record
from buttermilk.tools.fetch import Fetch


@pytest.fixture
def mock_record():
    """Create a mock record for testing."""
    return Record(
        record_id="test_id",
        title="Test Title",
        fulltext="Test content",
        paragraphs=["Test paragraph"],
        metadata={"source": "test"},
    )


@pytest.fixture
def fetch_agent():
    """Create a fetch agent for testing."""
    with patch("asyncio.create_task"):
        agent = Fetch(
            id="test_fetch",
            name="Test Fetch",
            description="Test fetch agent",
        )
        # Mock the data task to avoid actual loading
        agent._data_task = MagicMock()
        agent._data_task.done.return_value = True
        return agent


@pytest.mark.asyncio
async def test_fetch_process(fetch_agent, mock_record):
    """Test the process method of the fetch agent."""
    input_data = AgentInput(
        agent_id="test",
        content="fetch this",
        inputs={"record_id": "test_id"},
    )

    with patch.object(fetch_agent, "_run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_record
        result = await fetch_agent._process(input_data)

    mock_run.assert_called_once_with(record_id="test_id")
    assert isinstance(result, AgentOutput)
    assert result.agent_id == "test_fetch"
    assert result.content == "Test content"
    assert len(result.records) == 1
    assert result.records[0] == mock_record


@pytest.mark.asyncio
async def test_run_with_record_id(fetch_agent, mock_record):
    """Test the _run method with a record ID."""
    with patch.object(fetch_agent, "get_record_dataset", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_record
        result = await fetch_agent._run(record_id="test_id")

    mock_get.assert_called_once_with("test_id")
    assert result == mock_record


@pytest.mark.asyncio
async def test_run_with_uri(fetch_agent, mock_record):
    """Test the _run method with a URI."""
    with patch("buttermilk.tools.fetch.download_and_convert", new_callable=AsyncMock) as mock_download:
        mock_download.return_value = mock_record
        result = await fetch_agent._run(uri="https://example.com")

    mock_download.assert_called_once_with("https://example.com")
    assert result == mock_record


@pytest.mark.asyncio
async def test_run_with_prompt_containing_url(fetch_agent, mock_record):
    """Test the _run method with a prompt containing a URL."""
    with patch("buttermilk.tools.fetch.extract_url") as mock_extract:
        mock_extract.return_value = "https://example.com"
        with patch("buttermilk.tools.fetch.download_and_convert", new_callable=AsyncMock) as mock_download:
            mock_download.return_value = mock_record
            result = await fetch_agent._run(prompt="Check this https://example.com please")

    mock_extract.assert_called_once()
    mock_download.assert_called_once_with("https://example.com")
    assert result == mock_record


@pytest.mark.asyncio
async def test_run_with_prompt_containing_record_id(fetch_agent, mock_record):
    """Test the _run method with a prompt containing a record ID."""
    with patch("buttermilk.tools.fetch.extract_url") as mock_extract:
        mock_extract.return_value = None  # No URL found
        with patch.object(fetch_agent, "get_record_dataset", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_record
            result = await fetch_agent._run(prompt="!test_id")

    mock_extract.assert_called_once()
    mock_get.assert_called_once_with("test_id")
    assert result == mock_record


@pytest.mark.asyncio
async def test_get_record_dataset(fetch_agent, mock_record):
    """Test retrieving a record from a dataset."""
    # Create a sample dataframe with one matching record
    df = pd.DataFrame([{
        "record_id": "test_id",
        "title": "Test Title",
        "fulltext": "Test content",
        "paragraphs": ["Test paragraph"],
        "metadata": {"source": "test"},
    }])

    # Mock the data
    fetch_agent._data = {"test_dataset": df}

    result = await fetch_agent.get_record_dataset("test_id")

    assert result.record_id == "test_id"
    assert result.title == "Test Title"
    assert result.fulltext == "Test content"


@pytest.mark.asyncio
async def test_get_record_dataset_not_found(fetch_agent):
    """Test behavior when record is not found."""
    # Create a sample dataframe with no matching record
    df = pd.DataFrame([{
        "record_id": "other_id",
        "title": "Other Title",
        "fulltext": "Other content",
        "paragraphs": ["Other paragraph"],
        "metadata": {"source": "other"},
    }])

    # Mock the data
    fetch_agent._data = {"test_dataset": df}

    result = await fetch_agent.get_record_dataset("test_id")

    assert result is None


@pytest.mark.asyncio
async def test_receive_output_with_url(fetch_agent, mock_record):
    """Test receive_output method with a message containing a URL."""
    message = UserInstructions(content="<https://example.com>")

    with patch("buttermilk.tools.fetch.download_and_convert", new_callable=AsyncMock) as mock_download:
        mock_download.return_value = mock_record
        result = await fetch_agent.receive_output(message)

    mock_download.assert_called_once_with(uri="https://example.com")
    assert isinstance(result, AgentOutput)
    assert result.agent_id == "test_fetch"
    assert result.content == "Test content"
    assert len(result.records) == 1


@pytest.mark.asyncio
async def test_receive_output_with_record_id(fetch_agent, mock_record):
    """Test receive_output method with a message containing a record ID."""
    message = UserInstructions(content="!test_id")

    with patch.object(fetch_agent, "get_record_dataset", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_record
        result = await fetch_agent.receive_output(message)

    mock_get.assert_called_once_with("test_id")
    assert isinstance(result, AgentOutput)
    assert result.agent_id == "test_fetch"
    assert result.content == "Test content"
    assert len(result.records) == 1


@pytest.mark.asyncio
async def test_receive_output_no_match(fetch_agent):
    """Test receive_output method with a message that doesn't match patterns."""
    message = UserInstructions(content="Just a regular message")

    result = await fetch_agent.receive_output(message)

    assert result is None
