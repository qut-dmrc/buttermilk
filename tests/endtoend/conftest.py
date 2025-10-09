"""Configuration for end-to-end tests.

This file automatically applies markers to all tests in the directory.
"""

import pytest
from hydra import compose, initialize

from buttermilk._core.config_bootstrap import bootstrap_session_with_config_async

# Apply markers to all tests in this directory
pytestmark = pytest.mark.endtoend


@pytest.mark.anyio
@pytest.fixture(scope="session")
async def local_bm():
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
    with initialize(version_base=None, config_path="../../buttermilk/conf"):
        cfg = compose(config_name="minimal")
    bm, resolved_conf = await bootstrap_session_with_config_async(config=cfg)
    return bm


@pytest.mark.anyio
@pytest.fixture(scope="session")
async def local_conf(local_bm):
    """Minimal configuration dictionary from minimal.yaml."""
    return local_bm.cfg
