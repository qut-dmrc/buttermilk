import datetime
import os
import regex as re
import sys

import google.auth
import google.auth.credentials
import pandas as pd
import pytest
from cloudpathlib import CloudPath
from google.auth.credentials import TokenState
from numpy import isin
from pytest import CaptureFixture
from huggingface_hub import login


class Test00Setup:
    def test_imports(self):
        pass

    def test_python_version(self):
        """Check that the Python version is 3.11 or higher."""
        assert sys.version_info >= (3, 11)

    def test_gcloud_credentials_adc(self):
        credentials, project_id = google.auth.default()
        assert credentials.token_state == TokenState.FRESH

    def test_gcloud_no_json_key(self):
        """Check that the JSON key is not set."""
        assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ

    def test_gcs_bucket_set(self):
        """Check that the GCP bucket is set."""
        assert "GCS_BUCKET" in os.environ
        assert os.environ["GCS_BUCKET"]

    def test_bigquery(self, bm):
        df = bm.bigquery.run_query("SELECT True")
        assert df.iloc[0, 0] is True

    @pytest.mark.parametrize(
        ["table", "schema"],
        [("dmrc-analysis.tests.indicator", "datatools/chains/schemas/indicator.json")],
    )
    def test_database(self, bm, table, schema):
        """Delete and recreate the test table"""
        from google.cloud.bigquery.table import Table, TableReference

        test_table = table
        test_schema = read_yaml(schema)
        ref = TableReference.from_string(table_id=table)
        new_table = Table(table_ref=ref, schema=test_schema)

        assert gc.bq.create_table(table=new_table, exists_ok=True)

    def test_save_dir(self, gc):
        assert "/runs/testing/" in gc.save_dir
        assert CloudPath(gc.save_dir)


    def test_hf_login(self):
        login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])