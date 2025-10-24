import os
import sys

import google.auth
import google.auth.credentials

from buttermilk import logger
from buttermilk._core.log import logger  # noqa


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
