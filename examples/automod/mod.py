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
from buttermilk.buttermilk import SessionInfo
from buttermilk.libs import (
    HFInferenceClient,
    Llama2ChatMod,
    hf_pipeline,
    replicatellama2,
    replicatellama3,
)
from buttermilk.exceptions import FatalError
from buttermilk.agents.lc import LC
from buttermilk._core.runner_types import Job, RecordInfo, AgentInfo
from buttermilk.runner.flow import ResultsSaver, run_flow
from buttermilk.runner.helpers import group_and_filter_jobs, load_data
from buttermilk.runner.runner import Consumer, ResultsCollector, TaskDistributor
from buttermilk.tools.metrics import Metriciser, Scorer
from buttermilk.utils import (
    col_mapping_hydra_to_local,
    find_key_string_pairs,
    make_serialisable,
)

from buttermilk.runner.orchestrator import MultiFlowOrchestrator
import datetime
import itertools
from itertools import cycle
from tempfile import NamedTemporaryFile
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
from buttermilk.utils.flows import col_mapping_hydra_to_pf
from buttermilk import BM, logger
from promptflow.tracing import trace, start_trace
from rich import print as rprint

import os

from hydra import initialize, compose
from omegaconf import OmegaConf

## Run with, e.g.: 
## ```bash
## python -m examples.automod.mod +data=drag +step=ordinary +save=bq
## ```


# cfg changes with multiple options selected at runtime, but
# we want to make sure that run_id (which informs the logging
# tags and save directories) does not change across processes.
global_run_id = SessionInfo.make_run_id()

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    global bm, logger, global_run_id

    # Load the main ButterMilk singleton instance
    # This takes care of credentials, save paths, and other defaults
    bm = BM(run_id=global_run_id, cfg=cfg)
    logger = bm.logger

    # Print config to console
    rprint(OmegaConf.to_container(bm.cfg, resolve=True))


    try:
        for step_cfg in cfg.step:

            # Create an orchestrator to conduct all combinations of jobs we want to run
            orchestrator = MultiFlowOrchestrator(step=step_cfg, data=cfg.data, save=cfg.save)
            step_name=step_cfg.name

            t0 = datetime.datetime.now()
            try:
                
                async def run():
                    with atqdm(colour='magenta', 
                                desc=step_name,
                                bar_format="{desc:20}: {bar:50} | {rate_inv_fmt}",
                            ) as pbar: 
                        main_task = asyncio.create_task(orchestrator.run())
                        while not main_task.done():
                            pbar.total = sum([orchestrator._tasks_remaining, orchestrator._tasks_completed, orchestrator._tasks_failed])
                            pbar.n = sum([orchestrator._tasks_completed, orchestrator._tasks_failed])
                            pbar.refresh()
                            await asyncio.sleep(1)

                asyncio.run(run())

            except KeyboardInterrupt:
                # we have been interrupted. Abort gracefully if possible -- the first time. The second time, abort immediately.
                logger.info(
                    "Keyboard interrupt. Quitting run."
                )
                break

            except Exception as e:
                logger.exception(dict(message=f"Failed step {step_name} on run {global_run_id} on {cfg.run.platform}", error=str(e)))

            finally:
                t1 = datetime.datetime.now()
                logger.info(f"Completed step {step_name} on run {global_run_id} in {format_timespan(t1-t0)}."
                    )

    except KeyboardInterrupt:
        # Second time; quit immediately.
        raise FatalError("Keyboard interrupt. Aborting immediately.")

    except Exception as e:
        logger.exception(dict(message=f"Failed run on {cfg.run.platform}", error=str(e)))
    pass

if __name__ == "__main__":
    main()
