import asyncio

from buttermilk import init_async
from buttermilk._core.agent import AgentInput
from buttermilk.agents import FetchAgent, LLMAgent

# Descriptive name for the job that will be used in logging, tracing, and saving
JOB = "rights_extract"


async def init_buttermilk():
    """Initialize Buttermilk environment using the PRIMARY async pathway.

    This is THE recommended way to initialize Buttermilk in async contexts.
    Returns the BM instance with config accessible via bm.cfg.

    Returns:
        BM instance ready to use
    """
    # PRIMARY async initialization - this is the recommended approach
    # Simple one-liner that handles all bootstrap logic internally
    bm = await init_async(job=JOB, project_name="dbr", overrides=["llms=full"])

    # Access config and logger from the BM instance
    # cfg = bm.cfg
    # logger = bm.logger

    bm.logger.info("Initialized Buttermilk", session_info=bm.session_info)

    return bm


async def run_flow(bm):
    """Main function to run the rights extraction process.

    This function runs the extraction using the configured agents.

    Args:
        bm: The Buttermilk instance with config accessible via bm.cfg
    """
    # Access config from BM instance
    cfg = bm.cfg
    logger = bm.logger

    fetch = FetchAgent(storage=None)
    dbr = LLMAgent(
        role="DBR",
        name="RightsExtractor",
        description="Expert analysts who can extract features from declarations and bills of rights",
        agent_obj=LLMAgent,
        num_runs=1,
        parameters={"template": "rights"},  # Fixed: was "dbr", should be "rights"
        variants={"model": cfg.llms.general},
    )
    uri = "https://www.adbusters.org/articles-coded/digital-bill-of-rights"
    record = await fetch.fetch_uri(uri)

    dbr_input = AgentInput(inputs={"record": record})
    features = await dbr.invoke(message=dbr_input)

    logger.info("Extracted features", features=features)


async def main():
    """Main entry point demonstrating proper async initialization."""
    # Initialize Buttermilk using init_async - THE recommended way
    bm = await init_async(job=JOB, project_name="dbr", overrides=["llms=full"])

    # Run the flow
    await run_flow(bm)

    # Graceful shutdown
    await bm.graceful_shutdown()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
