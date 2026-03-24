"""Configuration for end-to-end tests.

This file automatically applies markers to all tests in the directory.
"""

import pytest
from hydra import compose, initialize

from buttermilk._core.config_bootstrap import bootstrap_session_with_config_async

# Apply markers to all tests in this directory
# End-to-end tests need longer timeout (240s) than unit tests (60s default)
# Note: This must also be passed via --timeout=240 when using pytest-xdist (-n)
pytestmark = [pytest.mark.slow, pytest.mark.timeout(240)]


def pytest_addoption(parser):
    """Add custom command line options for test configuration."""
    parser.addoption(
        "--run-expensive",
        action="store_true",
        default=False,
        help="Run tests with expensive image generation models (DALLE, SD3, SDXL, SDXLReplicate)",
    )


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


@pytest.fixture(scope="session")
async def local_conf(local_bm):
    """Minimal configuration dictionary from minimal.yaml."""
    return local_bm.cfg


@pytest.fixture
def image_clients(request):
    """Returns image generation clients based on --run-expensive flag.

    By default, returns only cheap/fast models from the centralized registry:
    - VertexImagen3Fast (Imagen 3.0 Fast - GCP Vertex AI)
    - VertexImagen4Fast (Imagen 4.0 Fast - GCP Vertex AI)
    - SD35Large (Stable Diffusion 3.5 Large - Azure)
    - FLUX11Pro (FLUX 1.1 Pro - Azure)

    With --run-expensive flag, returns all models from registry including:
    - All Vertex Imagen variants (3.0, 4.0, 4.0 Ultra)
    - DALLE (OpenAI - expensive per image)
    - SD3 (Stability AI direct API)
    - SDXL (HuggingFace - base + refiner = 2 calls)
    - SDXLReplicate (Replicate API)
    - SD (Stable Diffusion 2.1 - Replicate)
    """
    from buttermilk.agents.imagegen import ALL_IMAGE_CLIENTS, CHEAP_IMAGE_CLIENTS

    if request.config.getoption("--run-expensive"):
        return ALL_IMAGE_CLIENTS
    else:
        return CHEAP_IMAGE_CLIENTS
