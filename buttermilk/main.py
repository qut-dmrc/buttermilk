##########
##
# Example script to bootstrap Buttermilk with real configuration using Hydra.
# This script demonstrates how to initialize the Buttermilk context and infrastructure.
##
##########

from buttermilk import logger
from buttermilk.utils import cli

# # Notebooks
# bm = nb.init(job="my_analysis", name="my_project")
# logger.info("Started analysis")

logger.info("=== Testing simple new session creation ===")

# First session
logger.info("1. Creating first session...")
bm1 = cli.init(job="first_analysis", name="project_alpha")
logger.info("First session started")
logger.info(f"   Session 1: {bm1.session_info.session_id}")
logger.info(f"   Name: {bm1.session_info.name}, Job: {bm1.session_info.job}")

# Second session - just call init again
logger.info("2. Creating second session...")
bm2 = cli.init(job="second_analysis", name="project_beta")
logger.info("Second session started")
logger.info(f"   Session 2: {bm2.session_info.session_id}")
logger.info(f"   Name: {bm2.session_info.name}, Job: {bm2.session_info.job}")

# Third session with same name but different job
logger.info("3. Creating third session (same name, different job)...")
bm3 = cli.init(job="third_analysis", name="project_alpha")
logger.info("Third session started")
logger.info(f"   Session 3: {bm3.session_info.session_id}")
logger.info(f"   Name: {bm3.session_info.name}, Job: {bm3.session_info.job}")

logger.info("✅ All sessions created successfully!")
logger.info("Each session reuses the same infrastructure but has its own context.")
