##########
##
# Example script to bootstrap Buttermilk with real configuration using Hydra.
# This script demonstrates how to initialize the Buttermilk context and infrastructure.
##
##########
import asyncio

from hydra import compose, initialize

from buttermilk import set_bm
from buttermilk._core.config_bootstrap import ConfigurationBootstrapper

# Load configuration
with initialize(version_base=None, config_path="buttermilk/conf"):
    cfg = compose(config_name="config")

# Create bootstrapper
bootstrapper = ConfigurationBootstrapper(config=cfg)

# Step 1: Bootstrap full context (ExecutionContext + Infrastructure)
execution_context, infrastructure = asyncio.run(bootstrapper.bootstrap_full_context())

# Step 2: Bootstrap session context using existing infrastructure
session_bm = asyncio.run(
    bootstrapper.bootstrap_session_context(
        name="your_session_name",
        job="your_job_name",
        infrastructure=infrastructure,  # Use existing infrastructure
    )
)

# Step 3: Set as global singleton (if needed)
set_bm(session_bm)

# Step 4. When needed, create new session, reusing existing infrastructure
new_session_bm = asyncio.run(
    bootstrapper.bootstrap_session_context(
        name="session2",
        job="job2",
        infrastructure=infrastructure,  # REUSE existing
    )
)
