###
# You can debug from here, or run from the command line with:
#   python examples/automod/pfmod.py --multirun hydra/launcher=joblib +experiments=trans judger.model=gpt4o,sonnet,llama31_70b judger.standard=trans_factored.jinja2,trans_tja.jinja2,trans_hrc.jinja2,trans_glaad.jinja2,trans_simplified.jinja2
###
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
from buttermilk.tools.metrics import Metriciser, Scorer
from buttermilk.utils import col_mapping_hydra_to_local

BASE_DIR = Path(__file__).absolute().parent
import datetime
import itertools
from itertools import cycle
from tempfile import NamedTemporaryFile

# # Submit the pipeline
# experiment = Experiment(workspace=ws, name='batch-flow-experiment')
# run = experiment.submit(pipeline)
# run.wait_for_completion(show_output=True)
from typing import Callable, Optional, Type, TypeVar

import cloudpathlib
import datasets
import torch
import tqdm

from buttermilk.toxicity import *
from buttermilk.utils.flows import col_mapping_hydra_to_pf
from buttermilk.utils.log import logger

# from azureml.core import Workspace, Experiment
# from azureml.pipeline.core import Pipeline, PipelineData

# # Initialize workspace
# ws = Workspace.from_config()

# # Define your data and processing steps here
# # ...

# # Create a pipeline
# pipeline = Pipeline(workspace=ws, steps=[...])

global_run_id = BM.make_run_id()
pflocal = LocalPFClient()
bm: BM = None

from promptflow.tracing import start_trace, trace

start_trace(resource_attributes={"run_id": global_run_id}, collection="automod")


def cache_data(uri: str) -> str:
    with NamedTemporaryFile(delete=False, suffix=".jsonl", mode="wb") as f:
        dataset = f.name
        data = cloudpathlib.CloudPath(uri).read_bytes()
        f.write(data)
    return dataset

def run_flow(*, flow: object, flow_cfg,  flow_name, run_name: str, dataset: str|pd.DataFrame, run_cfg, step_outputs: Optional[str|pd.DataFrame] = None, from_run: Optional[str] = None, column_mapping: dict[str,str], batch_id: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame()

    init_vars = OmegaConf.to_object(flow_cfg.init)

    if run_cfg.platform == "azure":
        df = exec_pf(flow=flow, run_name=run_name, flow_name=flow_name, dataset=dataset, column_mapping=column_mapping, run=from_run, init_vars=init_vars)
    else:
        num_runs = flow_cfg.get('num_runs', 1)
        df = exec_local(flow=flow, num_runs=num_runs, run_name=run_name, run=from_run, flow_name=flow_name, dataset=dataset, step_outputs=step_outputs, column_mapping=column_mapping, init_vars=init_vars, batch_id=batch_id)
    return df

def exec_pf(*, flow, run_name, flow_name, dataset, column_mapping, run, init_vars) -> pd.DataFrame:
    columns = col_mapping_hydra_to_pf(column_mapping)
    environment_variables = {"PF_WORKER_COUNT": "11", "PF_BATCH_METHOD": "fork", "PF_LOGGING_LEVEL":"CRITICAL", }
    task = pflocal.run(
            flow=flow,
            data=dataset,
            run=run,
            display_name=flow_name,
            init=init_vars,
            column_mapping=columns,
            stream=False,
            name=run_name,
            timeout=150,
            environment_variables=environment_variables,
        )

    details = pflocal.get_details(task.name)

    # Stack and rename columns to a predictable format
    inputs = details[[x for x in details.columns if x.startswith('inputs.')]]
    inputs.columns = [x.replace('inputs.', '') for x in inputs.columns]

    # id_cols = [x for x in ['record_id', 'line_number'] if x in inputs.columns]
    # df = inputs[id_cols]

    details = details.assign(inputs=inputs.to_dict(orient='records')).drop(columns=inputs.columns, errors='ignore')
    details["timestamp"] = pd.to_datetime(datetime.datetime.now())

    # Add a column with the step results
    flow_outputs = details[[x for x in details.columns if x.startswith('outputs.')]]
    flow_outputs.columns = [x.replace('outputs.', '') for x in flow_outputs.columns]
    details.loc[flow_outputs.index.values, flow_name] = flow_outputs.to_dict(orient='records')

    logger.info(
        f"Run {task.name} for {flow} completed with status {task.status}. URL: {task._portal_url}. Processed {details.shape[0]} results."
    )
    return details

def exec_local(
    *,
    num_runs=1,
    flow,
    flow_name,
    batch_id,
    run_name,
    run: Optional[str] = None,
    colour: str = "magenta",
    init_vars: dict = {},
    dataset: str|pd.DataFrame,
    step_outputs: Optional[str|pd.DataFrame] = None,
    column_mapping: dict[str, str]
) -> pd.DataFrame:

    results = []

    if isinstance(dataset, str):
        dataset = pd.read_json(dataset, orient="records", lines=True)
    if isinstance(step_outputs, str):
        step_outputs = pd.read_json(step_outputs, orient="records", lines=True)
    if step_outputs is not None and isinstance(step_outputs, pd.DataFrame):
        try:
            dataset = dataset.join(step_outputs)
        except:
            dataset = pd.concat([dataset, step_outputs], axis='columns')

    # convert column_mapping to work for our dataframe
    columns = col_mapping_hydra_to_local(column_mapping)
    rename_dict = {v: k for k, v in columns.items()}
    input_df = dataset.rename(columns=rename_dict)
    input_df = pd.concat(itertools.repeat(input_df, num_runs))

    runnable = flow(**init_vars)
    try:
        if isinstance(runnable, ToxicityModel):
            # Run  as batch
            for details in tqdm.tqdm(
                runnable.moderate_batch(input_df),
                total=num_runs*dataset.shape[0],
                colour=colour,
                desc=flow_name,
                bar_format="{desc:30}: {bar:20} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ):
                details["timestamp"] = pd.to_datetime(datetime.datetime.now())

                results.append(details)

        else:
            for idx, row in tqdm.tqdm(
                input_df.iterrows(),
                total=num_runs*dataset.shape[0],
                colour=colour,
                desc=run_name,
                bar_format="{desc:30}: {bar:20} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ):
                # Run the flow for this line of data
                input_vars = {k: v for k, v in  row.items() if k in columns.keys()}
                details = runnable(**input_vars)

                details["timestamp"] = pd.to_datetime(datetime.datetime.now())

                # add input details to  the result row
                for k, v in input_vars.items():
                    if k not in details:
                        details[k] = v

                for k, v in init_vars.items():
                    if k not in details:
                        details[k] = v

                results.append(details)
    finally:
        results = pd.DataFrame(results).reset_index(drop=False,names='line_number')

        # add input details to the result dataset
        relevant_input_cols = list(set(input_df.columns).intersection(columns.keys()) - set(results.keys()))
        inputs = input_df.drop_duplicates(subset='record_id').set_index('record_id')[relevant_input_cols]
        inputs = pd.merge(results, inputs, left_on='record_id',right_index=True,how='inner', suffixes=(None,'_exec'))
        inputs = pd.Series(inputs[columns.keys()].to_dict(orient='records'), index=inputs.index, name='inputs')
        results = results.assign(inputs=inputs).drop(columns=columns.keys(),errors='ignore')

        # add batch details to the result dataset
        relevant_batch_cols = [c for c in ["model", "process", "standard"] if c in results.columns and c not in batch_id.keys()]
        batch_columns = pd.json_normalize(itertools.repeat(batch_id, results.shape[0]))
        batch_columns = pd.concat([results[relevant_batch_cols].reset_index(drop=True), batch_columns], axis='columns')

        results = results.assign(run_info=batch_columns.to_dict(orient='records')).drop(columns=relevant_batch_cols)


        bm.save(results, basename='partial_flow')
        del flow
        torch.cuda.empty_cache()
        gc.collect()

    return results

def save_to_bigquery(results: pd.DataFrame, save_cfg):
    from buttermilk.utils.save import upload_rows
    destination = upload_rows(rows=results, schema=save_cfg.schema, dataset=save_cfg.dataset)
    logger.info(f'Saved {results.shape[0]} rows to {destination}.')


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    global bm, logger, global_run_id
    # cfg changes with multiple options selected at runtime, but
    # we want to make sure that run_id (which informs the logging
    # tags and save directories) does not change across processes.
    bm = BM(run_id=global_run_id, cfg=cfg)

    logger = bm.logger

    run_results = {}
    run_names = {}

    try:
        for step_name, step_cfg in cfg.experiments.items():
            t0 = datetime.datetime.now()
            df = pd.DataFrame()
            try:
                if step_cfg.flow == 'lc':
                    flow_obj = LC
                elif step_cfg.flow == 'moderate':
                    flow_obj=load_tox_flow
                elif step_cfg.flow == 'judger':
                    flow_obj = Judger
                elif step_cfg.flow == 'modsynth':
                    flow_obj = 'buttermilk/flows/modsynth'
                run_name, df = run(flow_cfg=step_cfg, flow_obj=flow_obj, flow_name=step_name, run_cfg=cfg.run,run_names=run_names)
                run_results[step_name] = df
                run_names[step_name] = run_name
                pass

            except KeyboardInterrupt:
                # we have been interrupted. Abort gracefully if possible -- the first time. The second time, abort immediately.
                logger.info(
                    "Keyboard interrupt. Quitting run."
                )

            except Exception as e:
                logger.exception(dict(message=f"Failed {step_name} running on {cfg.run.platform}", error=str(e)))
                break
            finally:
                uri=None
                t1 = datetime.datetime.now()
                if df.shape[0]>0:
                    uri = bm.save(df.reset_index(), basename=step_name)
                logger.info(
                    dict(message=f"Completed step: {step_name}, processed {df.shape[0]} results in {format_timespan(t1-t0)}. Saved to {uri}", results=uri)
                )

    except KeyboardInterrupt:
        # Second time; quit immediately.
        raise FatalError("Keyboard interrupt. Aborting immediately.")

    except Exception as e:
        logger.exception(dict(message=f"Failed run on {cfg.run.platform}", error=str(e)))
    pass


def run(*, flow_name, flow_cfg, flow_obj, run_cfg, run_names:dict={}):
    global bm
    logger = bm.logger
    df = pd.DataFrame()
    run = None
    data_file = None

    batch_id = dict(
        run_id=bm.run_id,
        step=flow_name,
        **run_cfg
    )
    if 'run' in flow_cfg.data:
        # Previous run data, if applicable
        if 'from_run' in flow_cfg.data:
            # Load an explicit run id
            run = flow_cfg.data.from_run
        else:
            # Load a run from this batch run
            run = run_names[flow_cfg.data.run]

        batch_id['from_run']=run

    if 'uri' in flow_cfg.data:
        # And/or dataset
        data_file = cache_data(flow_cfg.data.uri)
        batch_id['dataset']=flow_cfg.data.name

    run_name = "_".join([str(x) for x in list(batch_id.values())])

    logger.info(dict(message=f"Starting {flow_name} running on {run_cfg.platform} with batch id {batch_id}.", **batch_id))

    # Run flow
    df = run_flow(flow=flow_obj,
                            flow_cfg=flow_cfg,
                            flow_name= flow_name,
                            dataset=data_file,
                            run_name = run_name,
                            from_run=run,
                            column_mapping=OmegaConf.to_object(flow_cfg.data.columns),
                            run_cfg=run_cfg,
                            batch_id=batch_id )

    df.loc[:, 'run_id'] = bm.run_id
    run_names[flow_name] = run_name

    # if flow_cfg.save:
    #     # save to bigquery
    #     save_to_bigquery(df, save_cfg=flow_cfg.save)

    return run_name, df



if __name__ == "__main__":
    main()
