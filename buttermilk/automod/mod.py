###
# You can debug from here, or run from the command line with:
#   python examples/automod/mod.py +experiments=trans +data=tja
###
import asyncio
import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import cloudpathlib
import hydra
import pandas as pd
import regex as re
from humanfriendly import format_timespan
from omegaconf import DictConfig, OmegaConf
from pydantic import model_validator

from buttermilk import BM
from buttermilk.apis import (
    HFInferenceClient,
    Llama2ChatMod,
    hf_pipeline,
    replicatellama2,
    replicatellama3,
)
from buttermilk.exceptions import FatalError
from buttermilk.flows.judge.judge import Judger
from buttermilk.lc import LC
from buttermilk.runner._runner_types import Job, RecordInfo, RunInfo, StepInfo
from buttermilk.runner.flow import ResultsSaver, run_flow
from buttermilk.runner.helpers import load_data
from buttermilk.runner.runner import Consumer, ResultsCollector, TaskDistributor
from buttermilk.tools.metrics import Metriciser, Scorer
from buttermilk.utils import col_mapping_hydra_to_local

BASE_DIR = Path(__file__).absolute().parent
import datetime
import itertools
from itertools import cycle
from tempfile import NamedTemporaryFile
from typing import Any, AsyncGenerator, Callable, Optional, Self, Type, TypeVar

import cloudpathlib
import datasets
import torch
import tqdm

#from buttermilk.toxicity import *
from buttermilk.utils.flows import col_mapping_hydra_to_pf
from buttermilk.utils.log import logger

### We have a standard runner in datatools.runner that allows us to run
### a batch of jobs asynchronously. Consumers take a Job item from a queue,
### process it, and return it as a new Job item. The main method to
### overwrite in any subclass is `.process()`

class Moderator(Consumer):
    ## We inherit from the Consumer class and override the process method.
    ## This is a standard pattern for creating a new Consumer class.
    _client: Optional[LC] = None
    flow_obj: Any

    @model_validator(mode='after')
    def init(self) -> Self:
        self._client =  self.flow_obj(**self.init_vars)
        del self.flow_obj  # Don't keep the class around

        return self

    async def process(self, *, job: Job) -> Job:
        """ Take a Job, process it, and return a Job."""
        job = await run_flow(flow=self._client, job=job)
        job.step_info = self.step_info
        return job

async def run(cfg, step_name, step_cfg):
    """
    This function runs the jobs asynchronously, and returns a list of
    results.
    """

    # The Orchestrator runs the jobs asynchronously
    orchestrator = TaskDistributor()

    # First we create a collector to save the results.
    if cfg.save.destination == 'bq':
        # Save to bigquery
        collector = ResultsSaver(dataset=cfg.save.dataset, dest_schema=cfg.save.schema)
    else:
        # default to save to GCS
        collector = ResultsCollector()

    # Register this collector as the default output queue for all workers.
    orchestrator.register_collector(collector)

    consumers = []

    if step_name == 'moderate':
        flow_obj=load_tox_flow
    else:
        flow_obj = LC

    init_vars = OmegaConf.to_object(step_cfg.init)
    for model in step_cfg.model:
        init_vars['default_model'] = model
        moderator = Moderator(task_name=model, step_name=step_name,
                                flow_obj=flow_obj, concurrent=step_cfg.concurrent, init_vars=init_vars,
                                run_info=bm.run_info)
        consumers.append(moderator)


    for task in consumers:
        orchestrator.register_task(consumer=task)

        dataset = pd.DataFrame()
        fields = []

        source_list = []

        for src in cfg.data:
            fields.extend(src.columns.keys())
            dataset = load_data(src)
            source_list.append(src.name)

        for src in step_cfg.data:
            fields.extend(src.columns.keys())
            data = load_data(src)
            dataset = group_and_filter_prior_step(dataset, new_data=data, prior_step=src)
            source_list.append(src.name)

        # add index, but don't remove record_id form the columns
        dataset = dataset.set_index('record_id', drop=False)
        dataset = dataset[fields]

        for i in range(step_cfg.num_runs):
            for idx, row in dataset.sample(frac=1).iterrows():
                job = Job(record_id=idx,
                        inputs=row.to_dict(),
                        run_info=bm.run_info,
                        parameters=step_cfg.parameters,source=source_list,
                        )

                # Add each job to the queue
                orchestrator.add_job(task_name=task.task_name, job=job)


    # Run!
    results = await orchestrator.run()

    return results

def group_and_filter_prior_step(df, new_data: pd.DataFrame, prior_step, max_n=32):
    if prior_step.type != 'job':
        return pd.concat([df, new_data])

    # Add columns to group by to the index
    idx_cols = []
    for group in prior_step.group:
        try:
            grp, col = group.split('.')

            # extract the contents of the nested source column that
            # will form the new index
            exploded = pd.json_normalize(new_data[grp])

            # limit to n list items per dataframe row
            exploded = exploded.groupby(exploded.index).agg({
                col: lambda x: x.tolist()[:max_n]
            }).reset_index(level=1, drop=True)

            new_data.loc[:, col] = exploded
        except ValueError:
            col = group
            pass  # no group in this column definition
        idx_cols.append(col)

    idx_cols = [ c for c in idx_cols if c in df.columns]

    new_data = new_data.set_index(idx_cols)

    # Get the names of all the mapped fields for the new record
    mapped_names = list(prior_step.record.keys())

    # rename the mapped fields
    new_data = new_data.rename(columns={v:k for k, v in prior_step.record.items()})

    # put all the mapped columns into a dictionary in
    # a new field named after this previous job step
    new_data[prior_step.name] = new_data[mapped_names].to_dict(orient='records')

    # Now aggregate them by the existing index we used above,
    # ensuring we shuffle and pick a maximum of max_n
    mapped_cols = new_data.sample(frac=1).groupby(
        level=0)[prior_step.name].agg(
            lambda x: x.tolist()[:max_n])

    # Add the column to the source dataset
    df = pd.merge(df, mapped_cols, left_on=idx_cols, right_index=True)

    return df


global_run_id = BM.make_run_id()

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    global bm, logger, global_run_id
    # cfg changes with multiple options selected at runtime, but
    # we want to make sure that run_id (which informs the logging
    # tags and save directories) does not change across processes.
    bm = BM(run_id=global_run_id, cfg=cfg)

    logger = bm.logger

    try:
        for step_name, step_cfg in cfg.experiments.items():

            t0 = datetime.datetime.now()
            try:

                asyncio.run(run(cfg, step_name=step_name, step_cfg=step_cfg))

            except KeyboardInterrupt:
                # we have been interrupted. Abort gracefully if possible -- the first time. The second time, abort immediately.
                logger.info(
                    "Keyboard interrupt. Quitting run."
                )

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
