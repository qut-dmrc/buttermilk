"""TRUE end-to-end tests for GCS DataFrame upload functionality.

This test uses REAL components:
- Real Google Cloud Storage: Uses prosocial-dev bucket from testing.yaml
- Real upload_dataframe_json function: Tests actual GCS operations
- Real storage round-trip: Validates data survives upload and retrieval

NO mocks - tests use live GCS connections via the project's existing infrastructure.

This validates the complete GCS upload workflow from DataFrame to stored JSONL.
"""

import json

import google.cloud.storage.blob
import pandas as pd
import pytest
import shortuuid
from google.cloud import storage

from buttermilk.utils.save import upload_dataframe_json


@pytest.fixture
def gcs_test_path():
    """Generate unique GCS path for test isolation."""
    unique_id = shortuuid.uuid()
    return f"gs://prosocial-dev/testing/test_save_gcs/{unique_id}.jsonl"


@pytest.fixture
def gcs_client():
    """Provide real GCS client for test operations."""
    return storage.Client()


def cleanup_gcs_file(uri: str, client: storage.Client):
    """Helper to clean up test files from GCS."""
    try:
        blob = google.cloud.storage.blob.Blob.from_string(uri=uri, client=client)
        blob.delete()
    except Exception:
        # Ignore cleanup errors - file may not exist
        pass


def test_upload_dataframe_json_success(gcs_test_path, gcs_client, real_bm):
    """Test successful upload of a DataFrame to GCS as JSONL.

    Validates complete round-trip: DataFrame -> GCS -> verification.
    """
    # ARRANGE: Create test DataFrame with realistic data
    test_df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"],
        "col3": [1.5, 2.5, 3.5]
    })

    try:
        # ACT: Upload to real GCS
        result_uri = upload_dataframe_json(test_df, gcs_test_path)

        # ASSERT: Verify upload succeeded
        assert result_uri == gcs_test_path

        # ASSERT: Verify file exists in GCS
        blob = google.cloud.storage.blob.Blob.from_string(uri=gcs_test_path, client=gcs_client)
        assert blob.exists(), f"Blob should exist at {gcs_test_path}"

        # ASSERT: Verify content is correct
        downloaded_content = blob.download_as_text()
        lines = downloaded_content.strip().split("\n")
        assert len(lines) == 3, "Should have 3 JSONL rows"

        # ASSERT: Verify first row structure
        first_row = json.loads(lines[0])
        assert first_row["col1"] == 1
        assert first_row["col2"] == "a"
        assert first_row["col3"] == 1.5

    finally:
        # CLEANUP: Remove test file
        cleanup_gcs_file(gcs_test_path, gcs_client)


def test_upload_dataframe_json_empty_df(gcs_test_path, gcs_client, real_bm):
    """Test uploading an empty DataFrame creates placeholder file."""
    # ARRANGE: Create empty DataFrame
    empty_df = pd.DataFrame()

    try:
        # ACT: Upload empty DataFrame
        result_uri = upload_dataframe_json(empty_df, gcs_test_path)

        # ASSERT: Function returns URI
        assert result_uri == gcs_test_path

        # ASSERT: Placeholder file exists (function creates empty file)
        blob = google.cloud.storage.blob.Blob.from_string(uri=gcs_test_path, client=gcs_client)
        assert blob.exists(), "Empty DataFrame should create placeholder file"

        # ASSERT: File is empty
        content = blob.download_as_text()
        assert content == "", "Placeholder file should be empty"

    finally:
        # CLEANUP: Remove test file
        cleanup_gcs_file(gcs_test_path, gcs_client)


def test_upload_dataframe_json_invalid_data(gcs_test_path):
    """Test that TypeError is raised if data is not a DataFrame."""
    # ARRANGE: Create non-DataFrame data
    invalid_data = "not a dataframe"

    # ACT & ASSERT: Expect TypeError
    with pytest.raises(TypeError, match="Input `data` must be a Pandas DataFrame"):
        upload_dataframe_json(invalid_data, gcs_test_path)


def test_upload_dataframe_json_duplicate_columns(gcs_test_path, gcs_client, real_bm):
    """Test that duplicate columns are handled correctly.

    The upload_dataframe_json function should deduplicate columns using
    reset_index_and_dedup_columns before upload. This function resets the
    index (adding 'index' column) and renames duplicate columns.
    """
    # ARRANGE: Create DataFrame with duplicate columns
    test_df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])

    try:
        # ACT: Upload DataFrame with duplicate columns
        result_uri = upload_dataframe_json(test_df, gcs_test_path)

        # ASSERT: Upload succeeded
        assert result_uri == gcs_test_path

        # ASSERT: Verify file exists
        blob = google.cloud.storage.blob.Blob.from_string(uri=gcs_test_path, client=gcs_client)
        assert blob.exists(), f"Blob should exist at {gcs_test_path}"

        # ASSERT: Verify content has deduplicated columns
        downloaded_content = blob.download_as_text()
        row = json.loads(downloaded_content.strip())

        # Function resets index (adds 'index' column) and renames duplicates
        keys = list(row.keys())
        assert len(set(keys)) == len(keys), "All column names should be unique"

        # Verify the expected deduplication: index, a, b, a2 (or similar)
        assert "index" in keys, "Index should be reset and included"
        assert "a" in keys, "Original 'a' column should exist"
        assert "b" in keys, "Original 'b' column should exist"
        # The second 'a' should be renamed (could be 'a2', 'a_1', etc.)
        duplicate_cols = [k for k in keys if k.startswith("a") and k != "a"]
        assert len(duplicate_cols) >= 1, "Duplicate 'a' column should be renamed"

    finally:
        # CLEANUP: Remove test file
        cleanup_gcs_file(gcs_test_path, gcs_client)
