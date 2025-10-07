##########
##
# Test script to bootstrap Buttermilk with real configuration using Hydra.
# This script demonstrates how to initialize the Buttermilk context and infrastructure.
##
##########
import asyncio

from hydra import compose, initialize

from buttermilk import init
from buttermilk._core.config_bootstrap import ConfigurationBootstrapper


def test_short_form_cli():
    bm = init(job="test_cli", project_name="testing")
    assert bm is not None
    assert bm.cloud_manager is not None


def test_short_form_nb():
    from buttermilk.utils.nb import nb_init

    bm = nb_init(project_name="testing", job="test_nb")
    logger = bm.logger
    logger.debug("logging seems to work")
    assert bm.cloud_manager is not None


def test_init():
    # Load configuration
    with initialize(version_base=None, config_path="../../buttermilk/conf"):
        cfg = compose(config_name="config")

    # Create bootstrapper
    bootstrapper = ConfigurationBootstrapper(config=cfg)

    # Step 1: Bootstrap full context
    execution_context = asyncio.run(bootstrapper.bootstrap_full_context())

    # Step 2: Bootstrap session context using existing infrastructure
    session_bm = asyncio.run(
        bootstrapper.bootstrap_session_context(
            name="your_session_name",
            job="your_job_name",  # Use existing infrastructure
        )
    )
    assert session_bm is not None
    assert execution_context is not None
    assert session_bm.cloud_manager is execution_context.cloud_manager
    assert session_bm.secret_manager is execution_context.secret_manager
    assert session_bm.llms is execution_context.llms
    assert session_bm.query_runner is execution_context.query_runner

    # Step 4. When needed, create new session, reusing existing infrastructure
    new_session_bm = asyncio.run(
        bootstrapper.bootstrap_session_context(
            name="session2",
            job="job2",
        )
    )

    assert new_session_bm is not None
    assert new_session_bm is not session_bm  # Ensure it's a new instance
    assert new_session_bm.cloud_manager is session_bm.cloud_manager
    assert new_session_bm.secret_manager is session_bm.secret_manager
    assert new_session_bm.llms is session_bm.llms
    assert new_session_bm.query_runner is session_bm.query_runner
