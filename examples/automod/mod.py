###
# You can debug from here, or run from the command line with:
#   python examples/automod/mod.py +experiments=trans +data=tja
###
import asyncio
import datetime
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import cloudpathlib
import hydra
import pandas as pd
import regex as re
import shortuuid
from humanfriendly import format_timespan
from omegaconf import DictConfig, OmegaConf
from pydantic import Field, model_validator

from tqdm.asyncio import tqdm as atqdm
from buttermilk import BM
from buttermilk._core.config import Project
from buttermilk.bm import SessionInfo
from buttermilk.libs import (
    HFInferenceClient,
    Llama2ChatMod,
    hf_pipeline,
    replicatellama2,
    replicatellama3,
)
from buttermilk.exceptions import FatalError
from buttermilk.agents.lc import LC

from buttermilk.runner.orchestrator import MultiFlowOrchestrator
import datetime
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Mapping,
    Optional,
    Self,
    Sequence,
    Type,
    TypeVar,
)

import cloudpathlib
from buttermilk import BM, logger
from rich import print as rprint


from hydra import initialize, compose
from omegaconf import OmegaConf

from buttermilk.runner.runloop import run_tasks

## Run with, e.g.: 
## ```bash
## python -m examples.automod.mod +data=drag +step=ordinary +save=bq job=<description> source=<desc>
## ```


# cfg changes with multiple options selected at runtime, but
# we want to make sure that run_id (which informs the logging
# tags and save directories) does not change across processes.
global_run_id = SessionInfo.make_run_id()

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: Project) -> None:
    global bm, logger, global_run_id
    
    # Load the main ButterMilk singleton instance
    # This takes care of credentials, save paths, and other defaults
    bm = BM(cfg=cfg)
    logger = bm.logger
    
    async def run():
        _continue = True
        for step_cfg in cfg.flows:
            if _continue:

                # Create an orchestrator to conduct all combinations of jobs we want to run
                orchestrator = MultiFlowOrchestrator(flow=step_cfg, source=cfg.job)
                step_name=step_cfg.name
            
                with atqdm(colour='magenta', 
                        desc=step_name,
                    ) as pbar: 
                    main_task = asyncio.create_task(run_tasks(task_generator=orchestrator.make_tasks(), num_runs=step_cfg.num_runs, max_concurrency=cfg.run.max_concurrency))

                    while not main_task.done():
                        pbar.total = sum([orchestrator._tasks_remaining, orchestrator._tasks_completed, orchestrator._tasks_failed])
                        pbar.n = sum([orchestrator._tasks_completed, orchestrator._tasks_failed])
                        pbar.refresh()
                        await asyncio.sleep(1)

                # exiting here, check for clean exit
                pass
                # set _continue = False  if loop failed

    asyncio.run(run())


if __name__ == "__main__":
    main()
    pass