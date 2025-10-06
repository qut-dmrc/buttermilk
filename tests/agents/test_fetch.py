from unittest.mock import MagicMock, patch

import pytest
from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime, TypeSubscription

from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentInput
from buttermilk._core.exceptions import ProcessingError  # Added ProcessingError
from buttermilk._core.types import Record
from buttermilk.agents.fetch import FetchAgent

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
        ProcessingError,  # Expect ProcessingError for failed URI fetch
        AgentInput(
            inputs=dict(prompt="Check out https://example.com"),  # This URI will be mocked to fail
        ),
    ),
    (
        ProcessingError,  # Expect ProcessingError for failed record_id fetch
        AgentInput(
            inputs={"step": "testing", "prompt": "Get `#record123`"},  # This ID will be mocked to fail
        ),
    ),
]


class TestFetch:
    """Tests for Fetch agent methods."""

    @pytest.fixture
    def fetch(self):
        return FetchAgent(description="test only")

    @pytest.mark.skip(reason="load_data method removed from FetchAgent - data sources initialized in __init__")
    @pytest.mark.anyio
    async def test_load_data(self, fetch, real_bm):
        """Test the load_data method with unified storage API."""
        from buttermilk._core.storage_config import FileStorageConfig

        mock_storage = MagicMock()
        test_config = FileStorageConfig(
            type="file",
            path="test.jsonl",
            dataset_name="test_data"
        )
        fetch.data = {"test_data": test_config}

        # Mock the get_storage method using patch
        with patch.object(real_bm, "get_storage", return_value=mock_storage) as mock_get_storage:
            await fetch.load_data()

            # Assert get_storage was called with the right config
            mock_get_storage.assert_called_once_with(test_config)
            # Assert data sources were assigned correctly
            assert isinstance(fetch._data_sources, dict)
            assert "test_data" in fetch._data_sources
            assert fetch._data_sources["test_data"] == mock_storage

    @pytest.mark.anyio
    @patch("buttermilk.agents.fetch.download_and_convert")
    async def test_fetch_nonexistent_uri_raises_processing_error(self, mock_download_and_convert, fetch: FetchAgent):
        """Test fetch raises ProcessingError when a URI is not found."""
        mock_download_and_convert.return_value = None

        uri_to_test = "http://example.com/nonexistentpage"
        with pytest.raises(ProcessingError, match=f"Record not found for URI: {uri_to_test}"):
            await fetch.fetch_uri(uri=uri_to_test)

        mock_download_and_convert.assert_called_once_with(uri_to_test)

    @pytest.mark.anyio
    @patch("buttermilk.agents.fetch.download_and_convert")
    async def test_fetch_specific_url_raises_processing_error(self, mock_download_and_convert, fetch: FetchAgent):
        """Test fetch raises ProcessingError for a specific URI when not found."""
        mock_download_and_convert.return_value = None

        specific_uri = "https://www.abc.net.au/religion/catherine-llewellyn-gender-affirming-healthcare-for-trans-youth"
        with pytest.raises(ProcessingError, match=f"Record not found for URI: {specific_uri}"):
            await fetch.fetch_uri(uri=specific_uri)

        mock_download_and_convert.assert_called_once_with(specific_uri)

    @pytest.mark.anyio
    async def test_fetch_nonexistent_id_raises_processing_error(self, fetch: FetchAgent):
        """Test fetch raises ProcessingError when a record ID is not found."""
        record_id_to_test = "nonexistent_id_123"
        
        # Mock storage to return None for the record ID
        mock_storage = MagicMock()
        mock_storage.get_record_by_id.return_value = None
        fetch._data_sources = {"test_data": mock_storage}

        with pytest.raises(ProcessingError, match=f"Record not found for ID: {record_id_to_test}"):
            await fetch.fetch_record(record_id_to_test, "test_data")

    @pytest.mark.anyio
    @pytest.mark.integration
    @pytest.mark.parametrize(
        argvalues=NEWS_RECORDS,
        argnames=["id", "uri", "expected_mimetype", "expected_size"],
        ids=[x[0] for x in NEWS_RECORDS],
    )
    async def test_ingest_news(self, fetch: FetchAgent, id, uri, expected_mimetype, expected_size):
        media_obj = await fetch.fetch_uri(uri=uri)
        assert len(media_obj.content) == expected_size
        assert media_obj.metadata["fetch_source_uri"] == uri


@pytest.fixture
def fetch_agent_cfg() -> AgentConfig:
    return AgentConfig(
        id="testing",
        role="fetch",
        name="fetch",
        description="fetch stuff",
        data={
            "tja_train": {
                "type": "file",
                "name": "tja_train",
                "path": "gs://prosocial-dev/data/tja_train.jsonl",
                "index": ["record_id"],
            },
        },
    )


@pytest.mark.skip(reason="Test uses wrong agent architecture - FetchAgent doesn't have register method for Autogen runtime")
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
        mock_r.metadata = {"uri": uri}
        mock_r.text = f"Mock content for {uri}"
        mock_r.fulltext = f"Mock content for {uri}"
        return mock_r

    # Mock for _get_record_dataset to simulate fetch failure for specific record IDs
    async def mock_get_record_dataset_conditional(record_id_to_lookup: str, **kwargs):
        if record_id_to_lookup == "record123":  # Corresponds to "Get `#record123`"
            return None  # Simulate fetch failure for this ID
        # Fallback for other record_ids:
        mock_r = MagicMock(spec=Record)
        mock_r.record_id = record_id_to_lookup
        mock_r.text = f"Mock content for {record_id_to_lookup}"
        mock_r.fulltext = f"Mock content for {record_id_to_lookup}"
        return mock_r

    with patch("buttermilk.utils.media.download_and_convert", side_effect=mock_download_and_convert_conditional) as mock_d_and_c, \
         patch.object(FetchAgent, "fetch_record", side_effect=mock_get_record_dataset_conditional) as mock_get_rec_dataset:

        agent_id = await FetchAgent.register(
            runtime,
            DefaultTopicId().type,
            lambda: FetchAgent(**fetch_agent_cfg.model_dump()),
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

    await runtime.stop_when_idle()  # Ensure runtime is stopped in all cases
