"""Integration tests for setup and infrastructure verification.

These tests verify that external services and infrastructure are properly
configured and accessible.
"""

import os

import pytest
from huggingface_hub import login

from buttermilk import BM
from buttermilk.utils.utils import read_yaml


def test_bigquery(real_bm: BM):
    """Test BigQuery connectivity and basic query execution."""
    df = real_bm.run_query("SELECT True")
    assert df.iloc[0, 0]


@pytest.mark.parametrize(
    ["table", "schema"],
    [("prosocial-443205.testing.flow", "buttermilk/schemas/flow.json")],
)
def test_database(real_bm: BM, table, schema):
    """Delete and recreate the test table."""
    from pathlib import Path

    from google.cloud.bigquery.table import Table, TableReference

    # Resolve schema path relative to project root
    schema_path = Path(__file__).parent.parent.parent.parent / schema
    if not schema_path.exists():
        pytest.skip(f"Schema file not found: {schema_path}")

    test_schema = read_yaml(str(schema_path))
    ref = TableReference.from_string(table_id=table)
    new_table = Table(table_ref=ref, schema=test_schema)

    assert real_bm.bq.create_table(table=new_table, exists_ok=True)


def test_hf_login():
    """Test HuggingFace authentication."""
    # Skip if token not available in environment
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        pytest.skip("HUGGINGFACEHUB_API_TOKEN not set in environment")

    # Integration test must fail if token not properly configured
    login(token=token)
