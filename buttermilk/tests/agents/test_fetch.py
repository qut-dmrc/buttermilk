from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime, TypeSubscription

from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk._core.runner_types import Record
from buttermilk.tools.fetch import Fetch


class TestFetch:
    """Tests for Fetch agent methods."""

    @pytest.fixture
    def mock_fetch(self):
        # actually, don't.
        return Fetch(id="test_fetch", role="fetch", description="test only")
        """Create a mock Fetch instance without dealing with initialization."""
        with patch.object(Fetch, "__init__", return_value=None):
            agent = Fetch()
            agent.role = "test_step"
            agent.publish = AsyncMock()
            return agent

    @pytest.mark.anyio
    async def test_load_data(self, mock_fetch):
        """Test the load_data method."""
        mock_data = MagicMock()
        mock_config = MagicMock()
        mock_fetch.config = mock_config

        with patch(
            "buttermilk.runner.helpers.prepare_step_df",
            return_value=mock_data,
        ) as mock_prepare:
            await mock_fetch.load_data()

            # Assert prepare_step_df was called with the right args
            mock_prepare.assert_called_once_with(mock_config.data)
            # Assert data was assigned correctly
            assert mock_fetch._data == mock_data

    def test_get_record_success(self, mock_fetch):
        """Test get_record with a valid record ID."""
        # Setup mock DataFrame
        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.to_dict.return_value = {"record_id": "123", "text": "Sample text"}

        mock_df.query.return_value = MagicMock()
        mock_df.query.return_value.shape = (1, 2)  # 1 row
        mock_df.query.return_value.iloc.__getitem__.return_value = mock_row

        mock_fetch._data = mock_df

        # Execute
        result = mock_fetch.get_record("123")

        # Assert
        mock_df.query.assert_called_once_with("record_id==@record_id")
        assert isinstance(result, Record)
        assert result.record_id == "123"
        assert result.fulltext == "Sample text"

    def test_get_record_multiple_matches(self, mock_fetch):
        """Test get_record raises error with multiple records."""
        # Setup mock DataFrame
        mock_df = MagicMock()
        mock_df.query.return_value = MagicMock()
        mock_df.query.return_value.shape = (2, 2)  # 2 rows - multiple matches

        mock_fetch._data = mock_df

        # Execute & Assert
        with pytest.raises(ValueError, match="More than one record found"):
            mock_fetch.get_record("123")

    def test_get_record_data_not_loaded(self, mock_fetch):
        """Test get_record raises error when data not loaded."""
        # Setup: no data
        mock_fetch._data = None

        # Execute & Assert
        with pytest.raises(ValueError, match="Data not loaded yet"):
            mock_fetch.get_record("123")

    @pytest.mark.anyio
    @patch("buttermilk.utils.utils.extract_url")
    @patch("buttermilk.utils.media.download_and_convert")
    async def test_handle_urls_with_url(
        self,
        mock_download,
        mock_extract_url,
        mock_fetch,
    ):
        """Test handle_urls with a URL."""
        # Setup
        agent_input = AgentInput(
            agent_role="test_agent",
            content="Check out https://example.com",
            inputs={"step": "testing"},
            agent_id="test",
        )
        ctx = MagicMock()
        mock_extract_url.return_value = "https://example.com"
        mock_record = MagicMock(spec=Record)
        mock_record.fulltext = "Example page content"
        mock_download.return_value = mock_record

        # Execute
        result = await mock_fetch.handle_urls(agent_input, ctx)

        # Assert
        mock_extract_url.assert_called_once_with(agent_input.content)
        mock_download.assert_called_once_with(uri="https://example.com")
        mock_fetch.publish.assert_called_once()
        assert isinstance(result, AgentOutput)
        assert result.content == mock_record.fulltext
        assert result.outputs["step"] == "test_step"

    @pytest.mark.anyio
    @patch("buttermilk.utils.utils.extract_url")
    @patch("re.match")
    async def test_handle_urls_with_record_id(
        self,
        mock_re_match,
        mock_extract_url,
        mock_fetch,
    ):
        """Test handle_urls with a record ID."""
        # Setup
        agent_input = AgentInput(
            agent_role="test_agent",
            content="Get `#record123`",
            inputs={"step": "testing"},
            agent_id="test",
        )
        ctx = MagicMock()
        mock_extract_url.return_value = None  # No URL
        match_result = MagicMock()
        match_result.group.return_value = "record123"
        mock_re_match.return_value = match_result
        mock_record = MagicMock(spec=Record)
        mock_record.fulltext = "Record content"
        mock_fetch.get_record = MagicMock(return_value=mock_record)

        # Execute
        result = await mock_fetch.handle_urls(agent_input, ctx)

        # Assert
        mock_extract_url.assert_called_once_with(agent_input.content)
        mock_re_match.assert_called_once_with(r"`#([\s\w]+)`", agent_input.content)
        mock_fetch.get_record.assert_called_once_with("record123")
        mock_fetch.publish.assert_called_once()
        assert isinstance(result, AgentOutput)
        assert result.content == mock_record.fulltext
        assert result.outputs["step"] == "test_step"

    @pytest.mark.anyio
    @patch("buttermilk.utils.utils.extract_url")
    @patch("re.match")
    async def test_handle_urls_without_url_or_id(
        self,
        mock_re_match,
        mock_extract_url,
        mock_fetch,
    ):
        """Test handle_urls with neither URL nor record ID."""
        # Setup
        agent_input = AgentInput(
            agent_role="test_agent",
            content="Just a regular message",
            inputs={"step": "testing"},
            agent_id="test",
        )
        ctx = MagicMock()
        mock_extract_url.return_value = None  # No URL
        mock_re_match.return_value = None  # No record ID

        # Execute
        result = await mock_fetch.handle_urls(agent_input, ctx)

        # Assert
        mock_extract_url.assert_called_once_with(agent_input.content)
        mock_re_match.assert_called_once_with(r"`#([\s\w]+)`", agent_input.content)
        mock_fetch.publish.assert_not_called()
        assert result is None

    @pytest.mark.anyio
    @patch("buttermilk.utils.utils.extract_url")
    async def test_handle_urls_with_request_to_speak(
        self,
        mock_extract_url,
        mock_fetch,
    ):
        """Test handle_urls works with RequestToSpeak messages."""
        # Setup
        agent_input = AgentInput(
            agent_role="test_agent",
            content="Check out https://example.com",
            inputs={},
            agent_id="test",
        )
        ctx = MagicMock()
        mock_extract_url.return_value = "https://example.com"
        mock_record = MagicMock(spec=Record)
        mock_record.fulltext = "Example page content"

        with patch(
            "buttermilk.utils.media.download_and_convert",
            return_value=mock_record,
        ) as mock_download:
            # Execute
            result = await mock_fetch.handle_urls(agent_input, ctx)

            # Assert
            mock_extract_url.assert_called_once_with(agent_input.content)
            mock_download.assert_called_once_with(uri="https://example.com")
            mock_fetch.publish.assert_called_once()
            assert isinstance(result, AgentOutput)


@pytest.fixture
def fetch_agent_cfg():
    return Agent(
        agent="Fetch",
        id="testing",
        description="fetch stuff",
        data=[
            {
                "type": "file",
                "name": "tja_train",
                "path": "gs://prosocial-dev/data/tja_train.jsonl",
                "index": ["record_id"],
            },
        ],
    )


messages = [
    (
        None,
        AgentInput(
            agent_role="test_agent",
            content="Just a regular message",
            inputs={"step": "testing"},
            agent_id="test",
        ),
    ),
    (
        "error",
        AgentInput(
            agent_role="test_agent",
            content="Check out https://example.com",
            inputs={},
            agent_id="test",
        ),
    ),
    (
        "missing",
        AgentInput(
            agent_role="test_agent",
            content="Get `#record123`",
            inputs={"step": "testing"},
            agent_id="test",
        ),
    ),
]


@pytest.mark.anyio
@pytest.mark.parametrize(["expected", "agent_input"], messages)
async def test_run_record_agent(
    fetch_agent_cfg,
    expected,
    agent_input,
):
    runtime = SingleThreadedAgentRuntime()
    agent_id = await Fetch.register(
        runtime,
        DefaultTopicId().type,
        lambda: Fetch(config=fetch_agent_cfg),
    )
    await runtime.add_subscription(
        TypeSubscription(
            topic_type=DefaultTopicId().type,
            agent_type=agent_id.type,
        ),
    )
    runtime.start()

    result = await runtime.send_message(
        agent_input,
        await runtime.get("default"),
    )
    await runtime.stop_when_idle()
    assert result == expected
