"""Configuration for end-to-end tests.

This file automatically applies markers to all tests in the directory.
"""

import pytest
from hydra import compose, initialize

from buttermilk._core.bm_init import BM
from buttermilk._core.config_bootstrap import bootstrap_session_with_config

# Apply markers to all tests in this directory
pytestmark = pytest.mark.endtoend


def init_local():
    """Minimal BM instance created from minimal.yaml configuration without cloud dependencies."""
    with initialize(version_base=None, config_path="../../buttermilk/conf"):
        cfg = compose(config_name="minimal")
    bm, resolved_conf = bootstrap_session_with_config(config=cfg)
    return bm, resolved_conf


local_bm_instance, local_conf_instance = init_local()


@pytest.fixture(scope="session")
def local_conf():
    """Minimal configuration dictionary from minimal.yaml."""
    return local_conf_instance


@pytest.fixture(scope="session")
def local_bm():
    """Minimal BM instance created from minimal.yaml configuration.

    This fixture provides a BM instance that doesn't require:
    - GCP credentials
    - Cloud logging
    - Secret Manager access
    - LLM API keys

    Useful for testing local-only functionality like:
    - Configuration loading
    - Record handling
    - Template rendering
    - Local storage operations
    """
    return local_bm_instance

