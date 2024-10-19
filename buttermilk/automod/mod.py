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

from buttermilk import BM
from buttermilk.apis import (
    HFInferenceClient,
    Llama2ChatMod,
    hf_pipeline,
    replicatellama2,
    replicatellama3,
)
from buttermilk.exceptions import FatalError
from buttermilk.lc import LC
from buttermilk.runner._runner_types import Job, RecordInfo, RunInfo, StepInfo
from buttermilk.runner.flow import ResultsSaver, run_flow
from buttermilk.runner.helpers import load_data
from buttermilk.runner.runner import Consumer, ResultsCollector, TaskDistributor
from buttermilk.tools.metrics import Metriciser, Scorer
from buttermilk.utils import (
    col_mapping_hydra_to_local,
    find_key_string_pairs,
    make_serialisable,
)

BASE_DIR = Path(__file__).absolute().parent
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

class JobProcessor(Consumer):
    ## We inherit from the Consumer class and override the process method.
    ## This is a standard pattern for creating a new Consumer class.
    _client: Optional[LC] = None
    flow_obj: Any

    @model_validator(mode='after')
    def init(self) -> Self:
        self._client = self.flow_obj(**self.init_vars)
        del self.flow_obj  # Don't keep the class around

        return self

    async def process(self, *, job: Job) -> Job:
        """ Take a Job, process it, and return a Job."""
        job = await run_flow(flow=self._client, job=job)
        job.step_info = self.step_info
        return job

async def run(cfg, step_cfg):
    """
    This function runs the jobs asynchronously, and returns a list of
    results.
    """

    step_name=step_cfg.name

    # The Orchestrator runs the jobs asynchronously
    orchestrator = TaskDistributor()

    # First we create a collector to save the results.
    if step_cfg.save.destination == 'bq':
        # Save to bigquery
        collector = ResultsSaver(dataset=step_cfg.save.dataset, dest_schema=step_cfg.save.schema)
    else:
        # default to save to GCS
        collector = ResultsCollector()

    # Register this collector as the default output queue for all workers.
    orchestrator.register_collector(collector)

    consumers = []

    if step_name == 'moderate':
        flow_obj=load_tox_flow
    elif step_name == 'metrics':
        flow_obj = Metriciser
    else:
        # Generic LLM chain based flow
        flow_obj = LC

    init_vars = OmegaConf.to_object(step_cfg.init)

    # Convert string values to single item lists
    string_vars = { k:[v] for k, v in init_vars.items() if isinstance(v, str) or not isinstance(v, Sequence) }
    init_vars.update(string_vars)

    # Generate all permutations of init_vars
    permutations = itertools.product(*(init_vars[key] for key in init_vars))

    # Convert the permutations to a list of dictionaries
    init_vars = [dict(zip(init_vars.keys(), values)) for values in permutations]

    for i, init_dict in enumerate(init_vars):
        processor = JobProcessor(agent=str(i), step_name=step_name,
                                flow_obj=flow_obj, init_vars=init_dict,
                                run_info=bm.run_info)
        consumers.append(processor)


    for processor in consumers:
        orchestrator.register_task(consumer=processor)

        dataset = pd.DataFrame()
        fields = []

        source_list = []
        dataset_configs = []

        # cfg.data is not ordered. Loop through and load the static data first.
        for data_id, src in cfg.data.items():
            if src.type == 'job':
                # end of list
                dataset_configs.append(src)
            else:
                # start of list
                dataset_configs = [src] + dataset_configs
            source_list.append(data_id)

        for src in dataset_configs:
            fields.extend(src.columns.keys())
            df = load_data(src)
            dataset = group_and_filter_prior_step(dataset, new_data=df, prior_step=src)
            

        # add index, but don't remove record_id form the columns
        dataset = dataset.reset_index().set_index('record_id', drop=False)[fields]

        for i in range(step_cfg.num_runs):
            for idx, row in dataset.sample(frac=1).iterrows():
                job = Job(record_id=idx,
                        inputs=make_serialisable(row.to_dict()),
                        run_info=bm.run_info,
                        source=source_list,
                        )

                # Add each job to the queue
                orchestrator.add_job(task_name=processor.agent, job=job)


    # Run!
    results = await orchestrator.run()

    return results

def group_and_filter_prior_step(df, new_data: pd.DataFrame, prior_step):
    if prior_step.type != 'job':
        new_data = new_data[prior_step.columns.keys()]
        return pd.concat([df, new_data])

    # expand and rename columns if we need to
    pairs_to_expand = list(find_key_string_pairs(prior_step.group)) + list(find_key_string_pairs(prior_step.columns))

    for col_name, group in pairs_to_expand:
        try:
            grp, col = group.split('.', 1)

            # extract the contents of the nested source column that
            # will form the new index
            try:
                exploded = pd.json_normalize(new_data[grp].apply(json.loads))
            except Exception as e:
                exploded = pd.json_normalize(new_data[grp])
                new_data.loc[:, col_name] = exploded
            else:
                new_data.loc[:, col_name] = exploded[col]
        except ValueError:
            pass  # no group in this column definition
            if col_name != group:
                # rename column
                new_data = new_data.rename(columns={group: col_name})

    # Add columns to group by to the index
    idx_cols = [ c for c in prior_step.group.keys() if c in new_data.columns]

    # Stack any nested fields in the mapping
    for k, v in prior_step.columns.items():
        if isinstance(v, Mapping):
            # put all the mapped columns into a dictionary in
            # a new field named as provided in the step config
            new_data.loc[:, k] = new_data[v.keys()].to_dict(orient='records')

    # Reduce down to n list items per index (but don't aggregate
    # at this time, just keep a random selection of rows)
    new_data = new_data.sample(frac=1).groupby(idx_cols).agg(
            lambda x: x.tolist()[:prior_step.max_records_per_group])




    # Add the column to the source dataset
    if df.shape[0]>0:
        # reset index columns that we're not matching on:
        group_only_cols = [x for x in idx_cols if x not in df.columns]
        idx_cols = list(set(idx_cols).difference(group_only_cols))
        new_data = new_data.reset_index(level=group_only_cols, drop=False)

        # Only return the columns we need
        new_data = new_data[prior_step.columns.keys()]

        df = pd.merge(df, new_data, left_on=idx_cols, right_index=True)
    else:
        # Only return the columns we need
        new_data = new_data[prior_step.columns.keys()]
        df = new_data.reset_index()


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
        for step_cfg in cfg.step:
            step_name=step_cfg.name

            t0 = datetime.datetime.now()
            try:

                asyncio.run(run(cfg, step_cfg=step_cfg))

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
