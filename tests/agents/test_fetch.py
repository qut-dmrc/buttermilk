from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buttermilk._core.agent import AgentConfig
from buttermilk._core.runner_types import RecordInfo
from buttermilk.agents.fetch import Fetch
from buttermilk.runner.chat import GroupChatMessage, InputRecord


@pytest.fixture
def mock_config():
    config = MagicMock(spec=AgentConfig)
    config.description = "Test Fetch Agent"
    config.name = "test_step"
    config.parameters = {}
    return config


@pytest.fixture
def mock_message_with_url():
    message = MagicMock(spec=GroupChatMessage)
    message.content = "Check out this link https://example.com/data"
    return message


@pytest.fixture
def mock_message_with_id():
    message = MagicMock(spec=GroupChatMessage)
    message.content = "Please fetch record `#record123`"
    return message


@pytest.fixture
def mock_message_without_url_or_id():
    message = MagicMock(spec=GroupChatMessage)
    message.content = "This is a regular message with no links or IDs"
    return message


@pytest.fixture
def mock_record():
    record = MagicMock(spec=RecordInfo)
    record.fulltext = "Content of the fetched document"
    return record


class TestFetchAgent:
    @pytest.mark.asyncio
    @patch("buttermilk.utils.utils.extract_url")
    @patch("buttermilk.utils.media.download_and_convert")
    async def test_handle_urls_with_url(
        self,
        mock_download,
        mock_extract_url,
        mock_config,
        mock_message_with_url,
        mock_record,
    ):
        # Setup
        mock_extract_url.return_value = "https://example.com/data"
        mock_download.return_value = mock_record

        fetch_agent = Fetch(config=mock_config)
        fetch_agent.publish = AsyncMock()

        # Execute
        ctx = MagicMock()
        result = await fetch_agent.handle_urls(mock_message_with_url, ctx)

        # Assert
        mock_extract_url.assert_called_once_with(mock_message_with_url.content)
        mock_download.assert_called_once_with(uri="https://example.com/data")
        fetch_agent.publish.assert_called_once()
        assert isinstance(result, InputRecord)
        assert result.content == mock_record.fulltext
        assert result.payload == mock_record
        assert result.step == "test_step"

    @pytest.mark.asyncio
    @patch("buttermilk.utils.utils.extract_url")
    @patch("re.match")
    async def test_handle_urls_with_record_id(
        self,
        mock_re_match,
        mock_extract_url,
        mock_config,
        mock_message_with_id,
        mock_record,
    ):
        # Setup
        mock_extract_url.return_value = None

        match_result = MagicMock()
        match_result.group.return_value = "record123"
        mock_re_match.return_value = match_result

        fetch_agent = Fetch(config=mock_config)
        fetch_agent.get_record = MagicMock(return_value=mock_record)
        fetch_agent.publish = AsyncMock()

        # Execute
        ctx = MagicMock()
        result = await fetch_agent.handle_urls(mock_message_with_id, ctx)

        # Assert
        mock_extract_url.assert_called_once_with(mock_message_with_id.content)
        mock_re_match.assert_called_once_with(
            r"`#([\s\w]+)`", mock_message_with_id.content
        )
        fetch_agent.get_record.assert_called_once_with("record123")
        fetch_agent.publish.assert_called_once()
        assert isinstance(result, InputRecord)
        assert result.content == mock_record.fulltext
        assert result.payload == mock_record
        assert result.step == "test_step"

    @pytest.mark.asyncio
    @patch("buttermilk.utils.utils.extract_url")
    @patch("re.match")
    async def test_handle_urls_without_url_or_id(
        self,
        mock_re_match,
        mock_extract_url,
        mock_config,
        mock_message_without_url_or_id,
    ):
        # Setup
        mock_extract_url.return_value = None
        mock_re_match.return_value = None

        fetch_agent = Fetch(config=mock_config)
        fetch_agent.publish = AsyncMock()

        # Execute
        ctx = MagicMock()
        result = await fetch_agent.handle_urls(mock_message_without_url_or_id, ctx)

        # Assert
        mock_extract_url.assert_called_once_with(mock_message_without_url_or_id.content)
        mock_re_match.assert_called_once_with(
            r"`#([\s\w]+)`", mock_message_without_url_or_id.content
        )
        fetch_agent.publish.assert_not_called()
        assert result is None

    @pytest.mark.asyncio
    @patch("buttermilk.runner.helpers.prepare_step_df")
    async def test_init_calls_prepare_step_df(self, mock_prepare_step_df, mock_config):
        # Setup
        mock_prepare_step_df.return_value = MagicMock()

        # Execute
        fetch_agent = Fetch(config=mock_config)

        # Assert
        mock_prepare_step_df.assert_called_once()
        assert fetch_agent._data == mock_prepare_step_df.return_value

    def test_get_record_with_single_match(self, mock_config):
        # Setup
        fetch_agent = Fetch(config=mock_config)

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.to_dict.return_value = {"id": "record123", "text": "Sample text"}

        mock_df.query.return_value = MagicMock()
        mock_df.query.return_value.shape = (1, 2)
        mock_df.query.return_value.iloc.__getitem__.return_value = mock_row

        fetch_agent._data = mock_df

        # Execute
        result = fetch_agent.get_record("record123")

        # Assert
        mock_df.query.assert_called_once_with("record_id==@record_id")
        assert isinstance(result, RecordInfo)

    def test_get_record_with_multiple_matches(self, mock_config):
        # Setup
        fetch_agent = Fetch(config=mock_config)

        mock_df = MagicMock()
        mock_df.query.return_value = MagicMock()
        mock_df.query.return_value.shape = (2, 2)  # More than one match

        fetch_agent._data = mock_df

        # Execute and Assert
        with pytest.raises(ValueError):
            fetch_agent.get_record("record123")
