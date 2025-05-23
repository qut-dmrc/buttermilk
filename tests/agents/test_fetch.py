from unittest.mock import MagicMock, patch

import pytest
from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime, TypeSubscription

from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentInput, AgentTrace
from buttermilk._core.exceptions import ProcessingError  # Added ProcessingError
from buttermilk._core.types import Record
from buttermilk.agents.fetch import FetchRecord

NEWS_RECORDS = [
    (
        "abc news web",
        "https://www.abc.net.au/news/2025-01-16/jewish-palestinian-australia-gaza/104825486",
        "text/html",
        5687,
    ),
    (
        "semaphor web",
        "https://www.semafor.com/article/11/12/2024/after-a-stinging-loss-democrats-debate-over-where-to-draw-the-line-on-transgender-rights",
        "text/html",
        5586,
    ),
]

messages = [
    (
        None,
        AgentInput(
            inputs={"step": "testing", "content": "Just a regular message"},
        ),
    ),
    (
        ProcessingError, # Expect ProcessingError for failed URI fetch
        AgentInput(
            inputs=dict(prompt="Check out https://example.com"), # This URI will be mocked to fail
        ),
    ),
    (
        ProcessingError, # Expect ProcessingError for failed record_id fetch
        AgentInput(
            inputs={"step": "testing", "prompt": "Get `#record123`"}, # This ID will be mocked to fail
        ),
    ),
]


class TestFetch:
    """Tests for Fetch agent methods."""

    @pytest.fixture
    def fetch(self):
        return FetchRecord(description="test only")

    @pytest.mark.anyio
    async def test_load_data(self, fetch):
        """Test the load_data method."""
        mock_data = MagicMock()
        mock_config = MagicMock()
        fetch.config = mock_config

        with patch(
            "buttermilk.runner.helpers.prepare_step_df",
            return_value=mock_data,
        ) as mock_prepare:
            await fetch.load_data()

            # Assert prepare_step_df was called with the right args
            mock_prepare.assert_called_once_with(mock_config.data)
            # Assert data was assigned correctly
            assert fetch._data == mock_data

    def test_get_record_success(self, fetch):
        """Test get_record with a valid record ID."""
        # Setup mock DataFrame
        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.to_dict.return_value = {"record_id": "123", "text": "Sample text"}

        mock_df.query.return_value = MagicMock()
        mock_df.query.return_value.shape = (1, 2)  # 1 row
        mock_df.query.return_value.iloc.__getitem__.return_value = mock_row

        fetch._data = mock_df

        # Execute
        result = fetch.get_record("123")

        # Assert
        mock_df.query.assert_called_once_with("record_id==@record_id")
        assert isinstance(result, Record)
        assert result.record_id == "123"
        assert result._text == "Sample text"

    def test_get_record_multiple_matches(self, fetch):
        """Test get_record raises error with multiple records."""
        # Setup mock DataFrame
        mock_df = MagicMock()
        mock_df.query.return_value = MagicMock()
        mock_df.query.return_value.shape = (2, 2)  # 2 rows - multiple matches

        fetch._data = mock_df

        # Execute & Assert
        with pytest.raises(ValueError, match="More than one record found"):
            fetch.get_record("123")

    def test_get_record_data_not_loaded(self, fetch):
        """Test get_record raises error when data not loaded."""
        # Setup: no data
        fetch._data = None

        # Execute & Assert
        with pytest.raises(ValueError, match="Data not loaded yet"):
            fetch.get_record("123")

    @pytest.mark.anyio
    @patch("buttermilk.utils.utils.extract_url")
    @patch("buttermilk.utils.media.download_and_convert")
    async def test_handle_urls_with_url(
        self,
        mock_download,
        mock_extract_url,
        fetch,
    ):
        """Test handle_urls with a URL."""
        # Setup
        agent_input = AgentInput(
            inputs={"step": "testing", "prompt": "Check out https://example.com"},
        )
        ctx = MagicMock()
        mock_extract_url.return_value = "https://example.com"
        mock_record = MagicMock(spec=Record)
        mock_record.fulltext = "Example page content"
        mock_download.return_value = mock_record

        # Execute
        result = await fetch.handle_urls(agent_input, ctx)

        # Assert
        mock_extract_url.assert_called_once_with(agent_input.inputs)
        mock_download.assert_called_once_with(uri="https://example.com")
        fetch.publish.assert_called_once()
        assert isinstance(result, AgentTrace)
        assert result.content == mock_record.fulltext
        assert result.outputs["step"] == "test_step"

    @pytest.mark.anyio
    @patch("buttermilk.utils.utils.extract_url")
    @patch("re.match")
    async def test_handle_urls_with_record_id(
        self,
        mock_re_match,
        mock_extract_url,
        fetch,
    ):
        """Test handle_urls with a record ID."""
        # Setup
        agent_input = AgentInput(
            inputs={"prompt": "Get `#record123`", "step": "testing"},
        )
        ctx = MagicMock()
        mock_extract_url.return_value = None  # No URL
        match_result = MagicMock()
        match_result.group.return_value = "record123"
        mock_re_match.return_value = match_result
        mock_record = MagicMock(spec=Record)
        mock_record.fulltext = "Record content"
        fetch.get_record = MagicMock(return_value=mock_record)

        # Execute
        result = await fetch.handle_urls(agent_input, ctx)

        # Assert
        mock_extract_url.assert_called_once_with(agent_input.inputs)
        mock_re_match.assert_called_once_with(r"`#([\s\w]+)`", agent_input.inputs)
        fetch.get_record.assert_called_once_with("record123")
        fetch.publish.assert_called_once()
        assert isinstance(result, AgentTrace)
        assert result.content == mock_record.fulltext
        assert result.outputs["step"] == "test_step"

    @pytest.mark.anyio
    @patch("buttermilk.utils.utils.extract_url")
    @patch("re.match")
    async def test_handle_urls_without_url_or_id(
        self,
        mock_re_match,
        mock_extract_url,
        fetch,
    ):
        """Test handle_urls with neither URL nor record ID."""
        # Setup
        agent_input = AgentInput(
            inputs={"prompt": "Just a regular message", "step": "testing"},
        )
        ctx = MagicMock()
        mock_extract_url.return_value = None  # No URL
        mock_re_match.return_value = None  # No record ID

        # Execute
        result = await fetch.handle_urls(agent_input, ctx)

        # Assert
        mock_extract_url.assert_called_once_with(agent_input.content)
        mock_re_match.assert_called_once_with(r"`#([\s\w]+)`", agent_input.content)
        fetch.publish.assert_not_called()
        assert result is None

    @pytest.mark.anyio
    @patch("buttermilk.utils.utils.extract_url")
    async def test_handle_urls_with_request_to_speak(
        self,
        mock_extract_url,
        fetch: FetchRecord,
    ):
        """Test handle_urls works with RequestToSpeak messages."""
        # Setup
        agent_input = AgentInput(
            inputs=dict(
                content="Check out https://example.com",
            ),
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
            result = await fetch.handle_urls(agent_input, ctx)

            # Assert
            mock_extract_url.assert_called_once_with(agent_input.inputs)
            mock_download.assert_called_once_with(uri="https://example.com")
            fetch.publish.assert_called_once()
            assert isinstance(result, AgentTrace)

    @pytest.mark.anyio
    @patch("buttermilk.utils.media.download_and_convert")
    async def test_fetch_nonexistent_uri_raises_processing_error(self, mock_download_and_convert, fetch: FetchRecord):
        """Test fetch raises ProcessingError when a URI is not found."""
        mock_download_and_convert.return_value = None

        uri_to_test = "http://example.com/nonexistentpage"
        with pytest.raises(ProcessingError, match=f"Record not found for URI: {uri_to_test}"):
            await fetch.fetch(uri=uri_to_test)

        mock_download_and_convert.assert_called_once_with(uri=uri_to_test)

    @pytest.mark.anyio
    @patch("buttermilk.utils.media.download_and_convert")
    async def test_fetch_specific_url_raises_processing_error(self, mock_download_and_convert, fetch: FetchRecord):
        """Test fetch raises ProcessingError for a specific URI when not found."""
        mock_download_and_convert.return_value = None

        specific_uri = "https://www.abc.net.au/religion/catherine-llewellyn-gender-affirming-healthcare-for-trans-youth"
        with pytest.raises(ProcessingError, match=f"Record not found for URI: {specific_uri}"):
            await fetch.fetch(uri=specific_uri)

        mock_download_and_convert.assert_called_once_with(uri=specific_uri)

    @pytest.mark.anyio
    async def test_fetch_nonexistent_id_raises_processing_error(self, fetch: FetchRecord):
        """Test fetch raises ProcessingError when a record ID is not found."""
        record_id_to_test = "nonexistent_id_123"

        async def mock_async_get_record_dataset(*args, **kwargs):
            # Simulate _get_record_dataset not finding the record
            return None

        with patch.object(fetch, "_get_record_dataset", new=mock_async_get_record_dataset) as patched_get_record_dataset:
            with pytest.raises(ProcessingError, match=f"Record not found for ID: {record_id_to_test}"):
                await fetch.fetch(record_id=record_id_to_test)
            patched_get_record_dataset.assert_called_once() # Verify it was called

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        argvalues=NEWS_RECORDS,
        argnames=["id", "uri", "expected_mimetype", "expected_size"],
        ids=[x[0] for x in NEWS_RECORDS],
    )
    async def test_ingest_news(self, fetch: FetchRecord, id, uri, expected_mimetype, expected_size):
        media_obj = await fetch._run(uri=uri)
        assert len(media_obj.text) == expected_size
        assert media_obj.uri == uri


@pytest.fixture
def fetch_agent_cfg() -> AgentConfig:
    return AgentConfig(
        id="testing",
        role="fetch",
        name="fetch",
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


@pytest.mark.parametrize(["expected", "agent_input"], messages)
@pytest.mark.anyio
async def test_run_record_agent(
    fetch_agent_cfg: AgentConfig,
    expected,
    agent_input: AgentInput,
):
    runtime = SingleThreadedAgentRuntime()

    # Mock for download_and_convert to simulate fetch failure for specific URIs
    async def mock_download_and_convert_conditional(uri: str, **kwargs):
        if uri == "https://example.com":
            return None  # Simulate fetch failure for this URI
        # Fallback for other URIs:
        mock_r = MagicMock(spec=Record)
        mock_r.uri = uri
        mock_r.text = f"Mock content for {uri}"
        mock_r.fulltext = f"Mock content for {uri}"
        return mock_r

    # Mock for _get_record_dataset to simulate fetch failure for specific record IDs
    async def mock_get_record_dataset_conditional(record_id_to_lookup: str, **kwargs):
        if record_id_to_lookup == "record123": # Corresponds to "Get `#record123`"
            return None # Simulate fetch failure for this ID
        # Fallback for other record_ids:
        mock_r = MagicMock(spec=Record)
        mock_r.record_id = record_id_to_lookup
        mock_r.text = f"Mock content for {record_id_to_lookup}"
        mock_r.fulltext = f"Mock content for {record_id_to_lookup}"
        return mock_r

    with patch("buttermilk.utils.media.download_and_convert", side_effect=mock_download_and_convert_conditional) as mock_d_and_c, \
         patch.object(FetchRecord, "_get_record_dataset", side_effect=mock_get_record_dataset_conditional) as mock_get_rec_dataset:

        agent_id = await FetchRecord.register(
            runtime, DefaultTopicId().type, lambda: FetchRecord(**fetch_agent_cfg.model_dump())
        )
        await runtime.add_subscription(
            TypeSubscription(
                topic_type=DefaultTopicId().type,
                agent_type=agent_id.type,
            ),
        )
        runtime.start()

        if expected is ProcessingError:
            # Determine the expected error message based on the input prompt
            prompt_str = agent_input.inputs.get("prompt", "")
            expected_match = ""
            if "https://example.com" in prompt_str:
                expected_match = "Record not found for URI: https://example.com"
            elif "Get `#record123`" in prompt_str:
                expected_match = "Record not found for ID: record123"

            with pytest.raises(ProcessingError, match=expected_match):
                await runtime.send_message(
                    agent_input,
                    await runtime.get("default"),  # topic
                )

            # Verify mocks were called if applicable
            if "https://example.com" in prompt_str:
                mock_d_and_c.assert_any_call(uri="https://example.com")
            elif "Get `#record123`" in prompt_str:
                # The prompt "Get `#record123`" will be parsed by fetch,
                # and `record_id` will become "record123" (after stripping `#` and ``).
                # The `fetch` method itself extracts "record123" from the prompt.
                mock_get_rec_dataset.assert_any_call("record123")
        else:
            result = await runtime.send_message(
                agent_input,
                await runtime.get("default"),  # topic
            )
            await runtime.stop_when_idle()
            assert result == expected

    await runtime.stop_when_idle() # Ensure runtime is stopped in all cases
