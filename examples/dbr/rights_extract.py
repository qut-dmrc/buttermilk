import asyncio
from pathlib import Path
from typing import Any

import hydra
from omegaconf import (
    OmegaConf,
)

from buttermilk import (
    BM,
    get_bm,  # Buttermilk manager singleton
    # dmrc as DMRC,
    logger,  # Centralized logger
    set_bm,
)
from buttermilk._core.agent import AgentInput
from buttermilk.agents import FetchAgent, LLMAgent

# Descriptive name for the job that will be used in logging, tracing, and saving
JOB = "rights_extract"


def init() -> Any:
    """Initializes the Buttermilk environment (`BM` instance).

    Add your own custom configuration overrides.

    This function should be called once at the beginning of a script or notebook.
    It sets up the global Buttermilk manager (`bm`) with the specified configuration.

    Args:
        overrides (list[str], optional): A list of Hydra override strings to customize
            the configuration. Defaults to [].
        path (str, optional): The path to the Hydra configuration directory. If not
            provided, it defaults to './conf'.

    Returns:
        Any: The fully instantiated configuration object from Hydra.
    """
    script_dir = Path(__file__).resolve().parent
    cfg_path = Path.cwd() / "conf"

    with hydra.initialize_config_dir(version_base=None, config_dir=cfg_path.as_posix(), job_name=JOB):
        # Start with a default config and apply user overrides
        overrides = [f"run.job={JOB}", "llms=full"]

        cfg = hydra.compose(config_name="config", overrides=overrides)
        OmegaConf.resolve(cfg)

        # Instantiate and set the global bm instance
        bm: BM = hydra.utils.instantiate(cfg.bm)
        set_bm(bm)

        logger.info(f"Initialized Buttermilk with run info: {bm.run_info}")

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

    logger.info(f"Extracted features: {features}")


if __name__ == "__main__":
    # Example usage of the init function
    cfg = init()

    asyncio.run(run_flow(cfg))
