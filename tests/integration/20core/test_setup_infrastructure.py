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
    [("prosocial-443205.testing.flow", "schemas/flow.json")],
)
def test_database(real_bm: BM, table, schema):
    """Delete and recreate the test table."""
    from google.cloud.bigquery.table import Table, TableReference

    test_schema = read_yaml(schema)
    ref = TableReference.from_string(table_id=table)
    new_table = Table(table_ref=ref, schema=test_schema)

    assert real_bm.bq.create_table(table=new_table, exists_ok=True)


def test_hf_login():
    """Test HuggingFace authentication."""
    # Integration test must fail if token not properly configured
    login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])