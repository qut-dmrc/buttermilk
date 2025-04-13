from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from google.api_core.exceptions import GoogleAPICallError

from buttermilk.utils.save import upload_dataframe_json

# Mock data
mock_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
mock_uri = "gs://test-bucket/test-file.jsonl"


@patch("google.cloud.storage.Client")
@patch("google.cloud.storage.blob.Blob.from_string")
def test_upload_dataframe_json_success(mock_blob_from_string, mock_storage_client):
    """Test successful upload of a DataFrame to GCS as JSONL."""
    mock_blob = MagicMock()
    mock_blob_from_string.return_value = mock_blob
    upload_dataframe_json(mock_df, mock_uri)
    mock_blob.upload_from_file.assert_called_once()

    # Assert that the data written to the blob is in the expected JSONL format
    call_args, _ = mock_blob.upload_from_file.call_args
    uploaded_data = call_args[0].read().decode("utf-8")
    expected_data = "\n".join(
        mock_df.to_json(orient="records", lines=True).splitlines()
    )
    assert uploaded_data == expected_data


@patch("google.cloud.storage.Client")
@patch("google.cloud.storage.blob.Blob.from_string")
def test_upload_dataframe_json_empty_df(mock_blob_from_string, mock_storage_client):
    """Test uploading an empty DataFrame."""
    empty_df = pd.DataFrame()
    result = upload_dataframe_json(empty_df, mock_uri)
    assert result == mock_uri
    mock_blob_from_string.assert_not_called()


@patch("google.cloud.storage.Client")
@patch("google.cloud.storage.blob.Blob.from_string")
def test_upload_dataframe_json_failure(mock_blob_from_string, mock_storage_client):
    """Test handling of GoogleAPICallError during upload."""
    mock_blob = MagicMock()
    mock_blob.upload_from_file.side_effect = GoogleAPICallError("Mock API error")
    mock_blob_from_string.return_value = mock_blob
    with patch("buttermilk.utils.save.logger.warning") as mock_logger_warning:
        try:
            upload_dataframe_json(mock_df, mock_uri)
        except:
            pass
        mock_logger_warning.assert_called_once()


def test_upload_dataframe_json_invalid_data():
    """Test that TypeError is raised if data is not a DataFrame."""
    with pytest.raises(TypeError):
        upload_dataframe_json("invalid data", mock_uri)


@patch("google.cloud.storage.Client")
@patch("google.cloud.storage.blob.Blob.from_string")
def test_upload_dataframe_json_duplicate_columns(
    mock_blob_from_string, mock_storage_client
):
    """Test that duplicate columns are handled correctly."""
    mock_blob = MagicMock()
    mock_blob_from_string.return_value = mock_blob
    df_with_duplicates = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df_with_duplicates = pd.concat([df_with_duplicates, df_with_duplicates], axis=1)
    upload_dataframe_json(df_with_duplicates, mock_uri)
    assert not any(df_with_duplicates.columns.duplicated())
    mock_blob.upload_from_file.assert_called_once()
