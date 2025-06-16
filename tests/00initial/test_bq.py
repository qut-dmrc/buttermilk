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
def writer():
    """Fixture to create a TableWriter instance for testing."""
    return TableWriter(
        table_path="test_project.test_dataset.test_table",
    )


def test_table_writer_init(writer):
    """Test that TableWriter initializes correctly."""
    assert writer.table_path == "test_project.test_dataset.test_table"
    # TableWriter should be initialized successfully
    assert writer.write_client is not None
    assert writer.stream == "_default"


@pytest.mark.anyio
@pytest.mark.integration
async def test_append_rows_integration(writer):
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
        FROM `{writer.table_path}`
    """
    query_job = client.query(query)
    results = query_job.result()
    assert len(list(results)) >= 2  # Assuming at least 2 rows were inserted
