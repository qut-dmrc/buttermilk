import asyncio
from typing import Any

from buttermilk import (
    get_bm,  # Buttermilk manager singleton
    logger,  # Centralized logger
)
from buttermilk._core.agent import AgentInput
from buttermilk.agents import FetchAgent, LLMAgent

# Descriptive name for the job that will be used in logging, tracing, and saving
JOB = "rights_extract"


def init() -> Any:
    """Initializes the Buttermilk environment (`BM` instance).

    This function should be called once at the beginning of a script or notebook.
    It sets up the global Buttermilk manager (`bm`) with the specified configuration.

    Returns:
        Any: The fully instantiated configuration object from Hydra.
    """
    from buttermilk import init
    
    # Simple initialization using the CLI utility with custom overrides
    bm = init(job=JOB, name="dbr_rights_extract", overrides=["llms=full"])
    
    logger.info("Initialized Buttermilk", session_info=bm.session_info)
    
    return cfg


async def run_flow(cfg):
    """Main function to run the rights extraction process.

    This function initializes the Buttermilk environment and starts the rights extraction
    process using the configured agents and flows.

    Args:
        cfg (Any): The configuration object containing Buttermilk settings.
    """
    bm = get_bm()
    fetch = FetchAgent(storage=None)
    dbr = LLMAgent(
        role="DBR",
        name="RightsExtractor",
        description="Expert analysts who can extract features from declarations and bills of rights",
        agent_obj=LLMAgent,
        num_runs=1,
        parameters={"template": "dbr"},
        variants={"model": cfg.llms.general},
    )
    uri = "https://www.adbusters.org/articles-coded/digital-bill-of-rights"
    record = fetch.fetch_uri(uri)

    dbr_input = AgentInput(inputs={"record": record})
    features = dbr.invoke(message=dbr_input)

    logger.info("Extracted features", features=features)


if __name__ == "__main__":
    # Example usage of the init function
    cfg = init()

    asyncio.run(run_flow(cfg))
