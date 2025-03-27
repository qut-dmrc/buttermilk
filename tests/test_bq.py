import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.cloud import bigquery_storage

from buttermilk.utils.bq import TableWriter

# Mock data for testing
MOCK_STREAM = "_default"
MOCK_ROWS = [
    {"test_time": datetime.datetime.now(), "success": True, "id": 1},
    {"test_time": datetime.datetime.now(), "success": True, "id": 2},
]


@pytest.fixture
def writer(flow):
    """Fixture to create a TableWriter instance for testing."""
    return TableWriter(
        table_path=flow.agents[0].save.dataset,
    )


@pytest.mark.anyio
async def test_append_rows(monkeypatch, writer):
    """Test appending rows to a BigQuery table."""
    # Mock the BigQueryWriteAsyncClient and its append_rows method
    mock_write_client = AsyncMock(spec=bigquery_storage.BigQueryWriteAsyncClient)
    mock_append_rows_stream = AsyncMock()
    mock_append_rows_stream.__aiter__.return_value = [
        MagicMock(),
        MagicMock(),
    ]  # Mock two responses
    mock_write_client.append_rows.return_value = mock_append_rows_stream
    monkeypatch.setattr(
        bigquery_storage,
        "BigQueryWriteAsyncClient",
        lambda: mock_write_client,
    )

    # Call the append_rows method
    await writer.append_rows(rows=MOCK_ROWS)

    # Assertions
    mock_write_client.append_rows.assert_called_once()
    request = mock_write_client.append_rows.call_args.args[0]

    # Add assertions to check if the proto_rows are correctly constructed
    assert request.proto_rows.rows.serialized_rows[0].proto_bytes == MOCK_ROWS[0]
    assert request.proto_rows.rows.serialized_rows[1].proto_bytes == MOCK_ROWS[1]

    # Since we mocked two responses, assert that the loop ran twice
    assert mock_append_rows_stream.__aiter__.call_count == 1


@pytest.mark.anyio
@pytest.mark.integration
async def test_append_rows_integration(writer, bm):
    """Test appending rows to a BigQuery table."""
    # Call the append_rows method
    results = await writer.append_rows(rows=MOCK_ROWS)
    assert all(results)

    # Add assertions to verify data is in the table
    # You'll need to use the BigQuery client to query the table and check if the data exists.
    # For example:
    from google.cloud import bigquery

    client = bigquery.Client()
    query = f"""
        SELECT *
        FROM `{bm.cfg.save.destination}`
    """
    query_job = client.query(query)
    results = query_job.result()
    assert len(list(results)) >= 2  # Assuming at least 2 rows were inserted
