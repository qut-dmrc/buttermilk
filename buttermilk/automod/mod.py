###
# You can debug from here, or run from the command line with:
#   python examples/automod/pfmod.py --multirun hydra/launcher=joblib +experiments=trans judger.model=gpt4o,sonnet,llama31_70b judger.standard=trans_factored.jinja2,trans_tja.jinja2,trans_hrc.jinja2,trans_glaad.jinja2,trans_simplified.jinja2
###
import asyncio
import datetime
import gc
import os
import resource
from math import pi
from multiprocessing import Process
from pathlib import Path
from random import shuffle
from tempfile import NamedTemporaryFile

import cloudpathlib
import hydra
import pandas as pd
import regex as re
from humanfriendly import format_timespan
from omegaconf import DictConfig, OmegaConf
from promptflow.azure import PFClient as AzurePFClient
from promptflow.client import PFClient as LocalPFClient
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
    init_vars: dict = {}
    step_info: StepInfo = {}
    run_info: RunInfo = {}

    @model_validator(mode='after')
    def init(self) -> Self:
        self._client =  LC(**self.init_vars)
        
    async def process(self, *, job: Job) -> AsyncGenerator[Job, Any]:
        """ Take a Job, process it, and return a Job."""
        try:
            job = await run_flow(flow=self._client, job=job)
        except Exception as e:
            job.error = f"Error running job: {e} {e.args=}"
        finally:
            job.timestamp = pd.to_datetime(datetime.datetime.now())
            job.step_info = self.step_info
            job.run_info = self.run_info
            return job

async def run(cfg):
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

    run_info = RunInfo(run_id=global_run_id, experiment_name="mod")
    for step_name, step_cfg in cfg.experiments.items():
        if step_name == 'moderate':
            flow_obj=load_tox_flow
        elif step_name == 'judge':
            flow_obj = LC
        elif step_name == 'synth':
            flow_obj = LC
        
        init_vars = OmegaConf.to_object(step_cfg.init)
        for model in step_cfg.model:
            init_vars['default_model'] = model
            moderator = Moderator(task_name=model, flow_obj=flow_obj, concurrent=1, init_vars=init_vars)
            consumers.append(moderator)

        data = {}

        # load data
        data['dataset'] = pd.read_json(step_cfg.data.uri, orient='records', lines=True)

        # load prior steps
        if 'jobs' in step_cfg:
            for prior_step in step_cfg.jobs:
                data[prior_step] = pd.read_json(step_cfg.data.uri, orient='records', lines=True)
        
        break # one at a time for now
    
    for task in consumers:
        orchestrator.register_task(consumer=task)

        step_info = StepInfo(agent_id=task.task_name, 
                      step_name=step_name, 
                      paramters=init_vars, 
                      source=step_cfg.data.name)
        # convert column_mapping to work for our dataframe
        columns = col_mapping_hydra_to_local(step_cfg.data.columns)
        fields = list(columns.keys())
        
        rename_dict = {v: k for k, v in columns.items()}
        
        dataset = pd.read_json(step_cfg.data.uri, orient='records', lines=True)
        dataset = dataset.rename(columns=rename_dict)
        
        # add additional inputs from groups of past jobs
        if 'jobs' in step_cfg:
            for prior_step in step_cfg.jobs:
                dataset = group_and_filter_prior_step(dataset, prior_step)
                fields.extend(list(prior_step.columns.keys()))

        # add index, but don't remove record_id form the columns
        dataset = dataset.set_index('record_id',drop=False)
        dataset = dataset[fields]

        for idx, row in dataset.sample(frac=1).iterrows():
            job = Job(record_id=idx,
                      step_info=step_info,
                      run_info=run_info,
                      inputs=row.to_dict(),
                      parameters=step_cfg.parameters,
                    )

            
            # Add each job to the queue
            orchestrator.add_job(task_name=task.task_name, job=job)


    # Run!
    results = await orchestrator.run()

    return results

def group_and_filter_prior_step(df, prior_step):
    if 'uri' in prior_step:
        new_data = pd.read_json(prior_step.uri, orient='records', lines=True)

    for group in prior_step.group:
        grp, col = group.split('.')

        # extract the contents of the nested source column that will form the
        # new index
        exploded = pd.json_normalize(new_data[grp])
        new_data.loc[:, col] = exploded[col]
        new_data = new_data.set_index(col)

    for mapped_name, col_name in prior_step.columns.items():
        new_data.groupby(level=0)[col_name].agg(list)
        
        mapped_col = new_data.groupby(level=0)[col_name].agg(list)
        mapped_col.name = mapped_name
        df = pd.merge(df, mapped_col, left_on=prior_step.group, right_index=True)
        
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
    
        t0 = datetime.datetime.now()
        try:

            asyncio.run(run(cfg))

        except KeyboardInterrupt:
            # we have been interrupted. Abort gracefully if possible -- the first time. The second time, abort immediately.
            logger.info(
                "Keyboard interrupt. Quitting run."
            )

        except Exception as e:
            logger.exception(dict(message=f"Failed run {global_run_id} on {cfg.run.platform}", error=str(e)))
            
        finally:
            t1 = datetime.datetime.now()
            logger.info(f"Completed run {global_run_id} in {format_timespan(t1-t0)}."
                )

    except KeyboardInterrupt:
        # Second time; quit immediately.
        raise FatalError("Keyboard interrupt. Aborting immediately.")

    except Exception as e:
        logger.exception(dict(message=f"Failed run on {cfg.run.platform}", error=str(e)))
    pass





if __name__ == "__main__":
    main()
