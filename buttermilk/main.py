##########
##
# Example script to bootstrap Buttermilk with real configuration using Hydra.
# This script demonstrates how to initialize the Buttermilk context and infrastructure.
##
##########

from buttermilk._core.config_bootstrap import init
from buttermilk._core.log import logger

# # When you need your own config
# bm = init(job="my_analysis", config_dir="./my_conf")

# # Notebooks
# bm = nb.init(job="my_analysis", name="my_project")
# logger.info("Started analysis")

logger.info("=== Testing simple new session creation ===")

# First session - project required
logger.info("1. Creating first session...")
bm1 = init(job="first_analysis", project_name="project_alpha")
logger.info("First session started")
logger.info(f"   Session 1: {bm1.session_info.session_id}")
logger.info(f"   Project: {bm1.session_info.project_name}, Job: {bm1.session_info.job}")
logger.info(f"   LLMs configured: {bm1.llms}")

# Second session - same project, different job (explicit project)
logger.info("2. Creating second session...")
bm2 = init(job="second_analysis")
logger.info("Second session started")
logger.info(f"   Session 2: {bm2.session_info.session_id}")
logger.info(f"   Project: {bm2.session_info.project_name}, Job: {bm2.session_info.job}")

logger.info("✅ All sessions created successfully!")
logger.info("Each session reuses the same infrastructure but has its own context.")
