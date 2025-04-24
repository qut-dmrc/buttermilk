import os
import sys

import google.auth
import google.auth.credentials
import pytest
from huggingface_hub import login

from buttermilk.bm import BM
from buttermilk.utils.utils import read_yaml


class Test00Setup:
    def test_imports(self):
        pass

    def test_python_version(self):
        """Check that the Python version is 3.10 or higher."""
        assert sys.version_info >= (3, 10)

    def test_gcloud_credentials_adc(self):
        credentials, project_id = google.auth.default()
        assert credentials
        assert project_id

    def test_gcloud_no_json_key(self):
        """Check that the JSON key is not set."""
        assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ

    def test_bigquery(self, bm: BM):
        df = bm.run_query("SELECT True")
        assert df.iloc[0, 0]

    @pytest.mark.parametrize(
        ["table", "schema"],
        [("prosocial-443205.testing.flow", "buttermilk/schemas/flow.json")],
    )
    def test_database(self, bm: BM, table, schema):
        """Delete and recreate the test table"""
        from google.cloud.bigquery.table import Table, TableReference

        test_schema = read_yaml(schema)
        ref = TableReference.from_string(table_id=table)
        new_table = Table(table_ref=ref, schema=test_schema)

        assert bm.bq.create_table(table=new_table, exists_ok=True)

    def test_hf_login(self):
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])
