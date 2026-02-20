"""Test QueryRunner converts numpy arrays from BigQuery ARRAY/REPEATED fields to Python lists.

This test validates that run_query() properly converts numpy arrays that BigQuery's
to_dataframe() returns for ARRAY/REPEATED fields into native Python lists.

Background:
- BigQuery stores ARRAY/REPEATED fields
- to_dataframe() returns these as numpy.ndarray objects
- Our code expects native Python lists
- The conversion should happen transparently in run_query()
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from google.cloud import bigquery

from buttermilk._core.query import QueryRunner


def test_run_query_converts_numpy_arrays_to_lists():
    """Test that run_query() converts numpy arrays in DataFrame to Python lists."""

    # Create mock BigQuery client with spec to pass Pydantic validation
    mock_bq_client = Mock(spec=bigquery.Client)

    # Create a DataFrame that simulates BigQuery's to_dataframe() output
    # ARRAY/REPEATED fields come back as numpy arrays
    mock_df = pd.DataFrame(
        {
            "call_id": ["test-1", "test-2"],
            "messages": [
                np.array(
                    [
                        '{"role":"user","content":"Hello"}',
                        '{"role":"assistant","content":"Hi"}',
                    ]
                ),
                np.array(['{"role":"user","content":"Goodbye"}']),
            ],
            "metadata": [{"key": "value1"}, {"key": "value2"}],
        }
    )

    # Mock the query job to return our test DataFrame
    mock_query_job = Mock()
    mock_query_job.result.return_value = Mock(total_rows=2)
    mock_query_job.to_dataframe.return_value = mock_df
    mock_query_job.total_bytes_billed = 1000
    mock_query_job.cache_hit = False
    mock_query_job.job_id = "test-job-123"

    mock_bq_client.query.return_value = mock_query_job

    # Create QueryRunner and execute query
    query_runner = QueryRunner(bq_client=mock_bq_client)
    result_df = query_runner.run_query("SELECT * FROM test_table")

    # ASSERT: Result should be a DataFrame
    assert isinstance(result_df, pd.DataFrame)

    # ASSERT: numpy arrays should be converted to Python lists
    assert isinstance(result_df.iloc[0]["messages"], list), (
        "messages field should be converted from numpy.ndarray to list"
    )
    assert isinstance(result_df.iloc[1]["messages"], list), (
        "messages field should be converted from numpy.ndarray to list"
    )

    # ASSERT: List contents should match
    assert len(result_df.iloc[0]["messages"]) == 2
    assert len(result_df.iloc[1]["messages"]) == 1

    # ASSERT: Other fields should remain unchanged
    assert result_df.iloc[0]["call_id"] == "test-1"
    assert result_df.iloc[0]["metadata"] == {"key": "value1"}


def test_run_query_handles_nested_numpy_arrays():
    """Test conversion of nested structures containing numpy arrays."""

    mock_bq_client = Mock(spec=bigquery.Client)

    # Create DataFrame with nested numpy arrays
    mock_df = pd.DataFrame(
        {
            "id": ["1"],
            "nested_arrays": [
                {"inner_list": np.array(["a", "b", "c"]), "regular_field": "value"}
            ],
        }
    )

    mock_query_job = Mock()
    mock_query_job.result.return_value = Mock(total_rows=1)
    mock_query_job.to_dataframe.return_value = mock_df
    mock_query_job.total_bytes_billed = 100
    mock_query_job.cache_hit = False
    mock_query_job.job_id = "test-job-456"

    mock_bq_client.query.return_value = mock_query_job

    query_runner = QueryRunner(bq_client=mock_bq_client)
    result_df = query_runner.run_query("SELECT * FROM test_table")

    # ASSERT: Nested numpy array should be converted
    nested_data = result_df.iloc[0]["nested_arrays"]
    assert isinstance(nested_data["inner_list"], list), (
        "Nested numpy arrays should be converted to lists"
    )
    assert nested_data["inner_list"] == ["a", "b", "c"]
    assert nested_data["regular_field"] == "value"


def test_run_query_without_numpy_arrays_unchanged():
    """Test that DataFrames without numpy arrays are unchanged."""

    mock_bq_client = Mock(spec=bigquery.Client)

    # Create DataFrame with only native Python types
    mock_df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "value": [100, 200],
            "tags": [["tag1", "tag2"], ["tag3"]],  # Already Python lists
        }
    )

    mock_query_job = Mock()
    mock_query_job.result.return_value = Mock(total_rows=2)
    mock_query_job.to_dataframe.return_value = mock_df
    mock_query_job.total_bytes_billed = 50
    mock_query_job.cache_hit = True
    mock_query_job.job_id = "test-job-789"

    mock_bq_client.query.return_value = mock_query_job

    query_runner = QueryRunner(bq_client=mock_bq_client)
    result_df = query_runner.run_query("SELECT * FROM test_table")

    # ASSERT: Data should be unchanged
    assert isinstance(result_df.iloc[0]["tags"], list)
    assert result_df.iloc[0]["tags"] == ["tag1", "tag2"]
    assert result_df.iloc[1]["value"] == 200


@pytest.mark.parametrize("return_df", [True, False])
def test_run_query_respects_return_df_parameter(return_df):
    """Test that return_df parameter is respected during conversion."""

    mock_bq_client = Mock(spec=bigquery.Client)
    mock_df = pd.DataFrame({"id": ["1"]})

    mock_query_job = Mock()
    mock_query_job.result.return_value = Mock(total_rows=1)
    mock_query_job.to_dataframe.return_value = mock_df
    mock_query_job.total_bytes_billed = 10
    mock_query_job.cache_hit = False
    mock_query_job.job_id = "test-job"

    mock_bq_client.query.return_value = mock_query_job

    query_runner = QueryRunner(bq_client=mock_bq_client)
    result = query_runner.run_query("SELECT * FROM test_table", return_df=return_df)

    if return_df:
        assert isinstance(result, pd.DataFrame)
    else:
        # Should return the query job object
        assert result == mock_query_job
