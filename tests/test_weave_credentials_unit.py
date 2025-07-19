"""Unit tests for weave credentials loading functionality.

This test module verifies that the BM class properly handles weave/WANDB credentials
from environment variables and secret manager according to issue #129.
"""

import os
import pytest
from unittest.mock import Mock, patch

from buttermilk._core.bm_init import BM


class TestWeaveCredentials:
    """Test weave credential loading from environment and secrets."""

    def setup_method(self):
        """Clear environment variables before each test."""
        for key in ['WANDB_API_KEY', 'WANDB_PROJECT', 'WANDB_ENTITY']:
            os.environ.pop(key, None)

    def test_weave_credentials_from_environment(self):
        """Test that weave credentials are loaded from environment variables."""
        # Set up environment variables
        os.environ['WANDB_API_KEY'] = 'test-api-key-from-env'
        os.environ['WANDB_PROJECT'] = 'test-project-from-env'
        os.environ['WANDB_ENTITY'] = 'test-entity-from-env'

        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('buttermilk._core.bm_init.SecretsManager'), \
             patch('weave.init') as mock_weave_init:

            mock_weave_init.return_value = Mock()

            bm = BM(
                name="test",
                job="test-env",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )

            # Mock empty credentials from secret manager
            bm._credentials_cached = {}

            # Access weave property to trigger credential setup
            weave_client = bm.weave

            # Verify weave.init was called with correct collection name
            mock_weave_init.assert_called_once_with("test-test-env")

            # Verify environment variables are preserved
            assert os.environ['WANDB_API_KEY'] == 'test-api-key-from-env'
            assert os.environ['WANDB_PROJECT'] == 'test-project-from-env'
            assert os.environ['WANDB_ENTITY'] == 'test-entity-from-env'

    def test_weave_credentials_from_secret_manager(self):
        """Test that weave credentials are loaded from secret manager when not in environment."""
        # Ensure environment variables are not set
        assert 'WANDB_API_KEY' not in os.environ
        assert 'WANDB_PROJECT' not in os.environ
        assert 'WANDB_ENTITY' not in os.environ

        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('buttermilk._core.bm_init.SecretsManager'), \
             patch('weave.init') as mock_weave_init:

            mock_weave_init.return_value = Mock()

            bm = BM(
                name="test",
                job="test-secrets",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )

            # Mock credentials from secret manager
            bm._credentials_cached = {
                'WANDB_API_KEY': 'test-api-key-from-secrets',
                'WANDB_PROJECT': 'test-project-from-secrets',
                'WANDB_ENTITY': 'test-entity-from-secrets'
            }

            # Access weave property to trigger credential setup
            weave_client = bm.weave

            # Verify weave.init was called
            mock_weave_init.assert_called_once_with("test-test-secrets")

            # Verify environment variables were set from secrets
            assert os.environ['WANDB_API_KEY'] == 'test-api-key-from-secrets'
            assert os.environ['WANDB_PROJECT'] == 'test-project-from-secrets'
            assert os.environ['WANDB_ENTITY'] == 'test-entity-from-secrets'

    def test_weave_credentials_partial_from_secrets(self):
        """Test loading partial credentials from secret manager when some are in environment."""
        # Set only API key in environment
        os.environ['WANDB_API_KEY'] = 'test-api-key-from-env'

        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('buttermilk._core.bm_init.SecretsManager'), \
             patch('weave.init') as mock_weave_init:

            mock_weave_init.return_value = Mock()

            bm = BM(
                name="test",
                job="test-partial",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )

            # Mock partial credentials from secret manager
            bm._credentials_cached = {
                'WANDB_PROJECT': 'test-project-from-secrets',
                'WANDB_ENTITY': 'test-entity-from-secrets'
            }

            # Access weave property to trigger credential setup
            weave_client = bm.weave

            # Verify weave.init was called
            mock_weave_init.assert_called_once_with("test-test-partial")

            # Verify environment API key is preserved
            assert os.environ['WANDB_API_KEY'] == 'test-api-key-from-env'
            # Verify project and entity were loaded from secrets
            assert os.environ['WANDB_PROJECT'] == 'test-project-from-secrets'
            assert os.environ['WANDB_ENTITY'] == 'test-entity-from-secrets'

    def test_weave_fallback_without_credentials(self):
        """Test weave fallback behavior when no credentials are available."""
        # Ensure no credentials in environment or secrets
        assert 'WANDB_API_KEY' not in os.environ
        assert 'WANDB_PROJECT' not in os.environ

        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('buttermilk._core.bm_init.SecretsManager'), \
             patch('weave.init') as mock_weave_init:

            # Make weave.init raise an exception to simulate auth failure
            mock_weave_init.side_effect = Exception("Authentication failed - no API key")

            bm = BM(
                name="test",
                job="test-fallback",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )

            # Mock empty credentials
            bm._credentials_cached = {}

            # Access weave property should not raise exception
            weave_client = bm.weave

            # Verify weave.init was attempted
            mock_weave_init.assert_called_once_with("test-test-fallback")

            # Verify we got a mock client instead of exception
            assert hasattr(weave_client, 'collection_name')
            assert weave_client.collection_name == "test-test-fallback"

            # Test mock client methods work without errors
            assert weave_client.create_call() is None
            weave_client.finish_call()  # Should not raise
            assert weave_client.get_call() is None

    def test_weave_credentials_secret_manager_error(self):
        """Test graceful handling when secret manager access fails."""
        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('buttermilk._core.bm_init.SecretsManager'), \
             patch('weave.init') as mock_weave_init:

            mock_weave_init.return_value = Mock()

            bm = BM(
                name="test",
                job="test-secret-error",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )

            # Mock secret manager access failure
            def mock_credentials_property():
                raise Exception("Secret manager access failed")
            
            type(bm).credentials = property(mock_credentials_property)

            # Access weave property should not raise exception
            weave_client = bm.weave

            # Verify weave.init was still called (with no credentials)
            mock_weave_init.assert_called_once_with("test-test-secret-error")

    def test_setup_weave_credentials_method_directly(self):
        """Test the _setup_weave_credentials method directly."""
        with patch('buttermilk._core.bm_init.CloudManager'), \
             patch('buttermilk._core.bm_init.SecretsManager'):

            bm = BM(
                name="test",
                job="test-direct",
                secret_provider={"type": "gcp", "project": "test-project"},
                clouds=[{"type": "gcp", "project": "test-project"}]
            )

            # Test with credentials available
            bm._credentials_cached = {
                'WANDB_API_KEY': 'direct-test-key',
                'WANDB_PROJECT': 'direct-test-project'
            }

            # Call method directly
            bm._setup_weave_credentials()

            # Verify environment variables were set
            assert os.environ['WANDB_API_KEY'] == 'direct-test-key'
            assert os.environ['WANDB_PROJECT'] == 'direct-test-project'

    def teardown_method(self):
        """Clean up environment variables after each test."""
        for key in ['WANDB_API_KEY', 'WANDB_PROJECT', 'WANDB_ENTITY']:
            os.environ.pop(key, None)