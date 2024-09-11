###
# You can debug from here, or run from the command line with:
#   python examples/automod/pfmod.py --multirun hydra/launcher=joblib +experiments=trans judger.model=gpt4o,sonnet,llama31_70b judger.standard=trans_factored.jinja2,trans_tja.jinja2,trans_hrc.jinja2,trans_glaad.jinja2,trans_simplified.jinja2
###
import os
import datetime
import re
import resource
from multiprocessing import Process
from pathlib import Path
from random import shuffle
from tempfile import NamedTemporaryFile
import gc
import cloudpathlib
import pandas as pd
from buttermilk import BM
from buttermilk.flows.evalqa.evalqa import EvalQA
from buttermilk.flows.judge.judge import Judger
from buttermilk.flows.results_bq import SaveResultsBQ
import hydra
from buttermilk.tools.metrics import Scorer, Metriciser
from buttermilk.utils.utils import read_text
from omegaconf import DictConfig, OmegaConf
from buttermilk.apis import (
    Llama2ChatMod,
    replicatellama2,
    replicatellama3,
    HFInferenceClient,
    HFLlama270bn,
    HFPipeline,
)
from humanfriendly import format_timespan
from buttermilk.utils import col_mapping_hydra_to_local
from promptflow.azure import PFClient as AzurePFClient
from promptflow.client import PFClient as LocalPFClient
BASE_DIR = Path(__file__).absolute().parent
import torch
import datetime
import tqdm
from itertools import cycle
from buttermilk.toxicity import *
from buttermilk.utils.log import getLogger

import cloudpathlib
from tempfile import NamedTemporaryFile
import itertools
# from azureml.core import Workspace, Experiment
# from azureml.pipeline.core import Pipeline, PipelineData

# # Initialize workspace
# ws = Workspace.from_config()

# # Define your data and processing steps here
# # ...

# # Create a pipeline
# pipeline = Pipeline(workspace=ws, steps=[...])

# # Submit the pipeline
# experiment = Experiment(workspace=ws, name='batch-flow-experiment')
# run = experiment.submit(pipeline)
# run.wait_for_completion(show_output=True)
from typing import Type, TypeVar, Callable, Optional
from buttermilk.utils.flows import col_mapping_hydra_to_pf

global_run_id = BM.make_run_id()
logger = None
pflocal = LocalPFClient()
bm: BM = None

def cache_data(uri: str) -> str:
    with NamedTemporaryFile(delete=False, suffix=".jsonl", mode="wb") as f:
        dataset = f.name
        data = cloudpathlib.CloudPath(uri).read_bytes()
        f.write(data)
    return dataset

def run_flow(*, flow: object, run_name: str, flow_name: str=None, dataset: str|pd.DataFrame, run_cfg, step_outputs: Optional[str|pd.DataFrame] = None, run_outputs: Optional[str] = None, column_mapping: dict[str,str], init_vars: dict = {}) -> pd.DataFrame:
    df = pd.DataFrame()
    if not flow_name:
        try:
            flow_name = flow.__name__.lower()
        except:
            flow_name = str(flow).lower()

    if run_cfg.platform == "azure":
        df = exec_pf(flow=flow, run_name=run_name, flow_name=flow_name, dataset=dataset, column_mapping=column_mapping, run=run_outputs, init_vars=init_vars)
    else:
        df = exec_local(flow=flow, run_name=run_name, flow_name=flow_name, dataset=dataset, step_outputs=step_outputs, column_mapping=column_mapping, init_vars=init_vars)
    return df

def exec_pf(*, flow, run_name, flow_name, dataset, column_mapping, run, init_vars) -> pd.DataFrame:
    logger.info(dict(message=f"Starting {flow_name} for run: {run_name}", **init_vars))
    columns = col_mapping_hydra_to_pf(column_mapping)
    environment_variables = {"PF_WORKER_COUNT": "11", "PF_BATCH_METHOD": "fork", "PF_LOGGING_LEVEL":"CRITICAL", "PF_DISABLE_TRACING": "true"}
    task = pflocal.run(
            flow=flow,
            data=dataset,
            run=run,
            display_name=flow_name,
            init=init_vars,
            column_mapping=columns,
            stream=False,
            name=run_name,
            timeout=60,
            environment_variables=environment_variables,
        )

    details = pflocal.get_details(task.name)

    # Stack and rename columns to a predictable format
    inputs = details[[x for x in details.columns if x.startswith('inputs.')]]
    inputs.columns = [x.replace('inputs.', '') for x in inputs.columns]

    id_cols = [x for x in ['record_id', 'line_number'] if x in inputs.columns]
    df = inputs[id_cols]

    # Add a column with the step results
    flow_outputs = details[[x for x in details.columns if x.startswith('outputs.')]]
    flow_outputs.columns = [x.replace('outputs.', '') for x in flow_outputs.columns]
    df.loc[flow_outputs.index.values, flow_name] = flow_outputs.to_dict(orient='records')

    # see if we can add groundtruth back in
    if 'groundtruth' in flow_outputs and 'groundtruth' not in df.columns:
        df.loc[flow_outputs.index.values, 'groundtruth'] = flow_outputs['groundtruth']

    logger.info(
        f"Run {task.name} for {flow} completed with status {task.status}. URL: {task._portal_url}. Processed {df.shape[0]} results."
    )
    return df

def exec_local(
    *,
    num_runs=1,
    flow,
    flow_name,
    run_name,
    colour: str = "magenta",
    init_vars: dict = {},
    dataset: str|pd.DataFrame,
    step_outputs: Optional[str|pd.DataFrame] = None,
    column_mapping: dict[str, str]
) -> pd.DataFrame:

    logger.info(dict(message=f"Starting {flow_name} running locally with flow {flow} with name: {run_name}"))
    t0 = datetime.datetime.now()

    results = []

    if isinstance(dataset, str):
        dataset = pd.read_json(dataset, orient="records", lines=True)
    if isinstance(step_outputs, str):
        step_outputs = pd.read_json(step_outputs, orient="records", lines=True)

    runs = itertools.product(range(num_runs), dataset.iterrows())

    for i, (idx, row) in tqdm.tqdm(
        runs,
        total=num_runs*dataset.shape[0],
        colour=colour,
        desc=run_name,
        bar_format="{desc:30}: {bar:20} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        runnable = flow(**init_vars)

        # convert column_mapping to work for our dataframe
        columns = col_mapping_hydra_to_local(column_mapping)
        input_vars = {}
        for key, mapped in columns.items():
            if step_outputs is not None and isinstance(step_outputs, pd.DataFrame) and mapped in step_outputs.columns:
                # Use the output from the previous step
                # This relies on the step_outputs keeping the same index as the dataset
                input_vars[key] = step_outputs.loc[idx, mapped]
            elif mapped in dataset.columns:
                input_vars[key] = row[mapped]
            else:
                ValueError("Unable to find data input for {key}: {mapped}")

        # Run the flow for this line of data
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

    results = pd.DataFrame(results)
    t1 = datetime.datetime.now()
    logger.info(
        f"Run {run_name} completed locally, processed {results.shape[0]} results in {format_timespan(t1-t0)}."
    )
    del flow
    torch.cuda.empty_cache()
    gc.collect()
    return results


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    global bm, logger, global_run_id
    # cfg changes with multiple options selected at runtime, but 
    # we want to make sure that run_id (which informs the logging
    # tags and save directories) does not change across processes.
    bm = BM(run_id=global_run_id, cfg=cfg)

    logger = bm.logger

    # from promptflow.tracing import start_trace, trace
    # start_trace(resource_attributes={"run_id": bm._run_id, "job": cfg.project.job}, collection=cfg.project.name)
    # Create a cycle iterator for the progress bar colours
    bar_colours = ["cyan", "yellow", "magenta", "green", "blue"]
    colour_cycle = cycle(bar_colours)
    if 'moderate' in cfg.experiments.keys():
        for model in cfg.experiments.moderate.models:
            cfg.experiments.moderate.init['model'] = model
            df = run(data=cfg.data, flow_cfg=cfg.experiments.moderate, flow_obj=load_tox_model, run_cfg=cfg.run)
    if 'judger' in cfg.experiments.keys():
        connections = bm._connections_azure
        cfg.experiments.judger.init['connections'] = connections
        df = run(data=cfg.data, flow_cfg=cfg.experiments.judger, flow_obj=Judger, evaluator_cfg=cfg.experiments.evaluator, run_cfg=cfg.run)
    pass

def run(*, data, flow_cfg, flow_obj, evaluator_cfg: Optional[dict]={}, run_cfg):
    global bm
    df = pd.DataFrame()
    data_file = cache_data(data.uri)

    # Run flow
    try:
        batch_id = dict(
            run_id=bm._run_id,
            step=flow_cfg.name,
            dataset=data.name,
            **run_cfg
        )
        for k,v in flow_cfg.init.items():
            if isinstance(v, str):
                # remove path extensions
                k = re.sub(r'_.*', '', str(k))
                v = re.sub(r'\..*', '', str(v))
                v = str.lower(v)
                batch_id[k] = v

        run_name = "_".join(list(batch_id.values()))
        flow_outputs = run_flow(flow=flow_obj,
                                flow_name=flow_cfg.name,
                                dataset=data_file,
                                run_name = run_name,
                                column_mapping=dict(data.columns),
                                init_vars=flow_cfg.init, run_cfg=run_cfg )

        # Set up  empty dataframe with batch details ready  to go
        df = pd.json_normalize(itertools.repeat(batch_id, flow_outputs.shape[0]))
        df = pd.concat([df, flow_outputs], axis='columns')

        # Run the evaulation flows
        for eval_model in evaluator_cfg.get("models", []):
            eval_name = f"{run_name}_evaluator_{eval_model}"
            init_vars = {"model": eval_model}
            flow_name = f'evaluator_{eval_model}'
            evals = run_flow(flow=EvalQA,
                            flow_name=flow_name,
                            dataset=data_file,
                            step_outputs = flow_outputs,
                            run_outputs = run_name,
                            run_name = eval_name,
                            column_mapping=dict(evaluator_cfg.columns),
                            init_vars = init_vars,
                            run_cfg=run_cfg
                    )

            # join the evaluation results
            try:
                df.loc[:, flow_name] = evals.to_dict(orient='records')
                if 'groundtruth' in evals and 'groundtruth' not in df.columns:
                    df.loc[:, 'groundtruth'] = evals['groundtruth']
            except Exception as e:
                # We might not get all the responses back. Try to join on line number instead?
                pass
                df = df.merge(evals[['line_number',flow_name]], left_on='line_number', right_on='line_number')

    finally:
        if df.shape[0]>0:
            uri = bm.save(df.reset_index(), basename='batch')
            logger.info(
                    msg=f"Completed batch: {batch_id} with {df.shape[0]} step results saved to {uri}.",
                    extra=dict(**batch_id,results=uri)
            )
    return df



if __name__ == "__main__":
    main()
