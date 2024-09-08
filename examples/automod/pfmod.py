###
# You can debug from here, or run from the command line with:
#   python examples/automod/pfmod.py --multirun hydra/launcher=joblib +experiments=trans judger.model=gpt4o,sonnet,llama31_70b judger.standard=trans_factored.jinja2,trans_tja.jinja2,trans_hrc.jinja2,trans_glaad.jinja2,trans_simplified.jinja2
###
import os
import datetime
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

logger = getLogger()
pflocal = LocalPFClient()
bm = BM()

def col_mapping_hydra_to_pf(mapping_dict: dict) -> dict:
    output = {}
    for k, v in mapping_dict.items():
        # need to escape the curly braces
        # prompt flow expects a mapping like:
        #   record_id: ${data.id}
        output[k] = f"${{{v}}}"

    return output


def cache_data(uri: str) -> str:
    with NamedTemporaryFile(delete=False, suffix=".jsonl", mode="wb") as f:
        dataset = f.name
        data = cloudpathlib.CloudPath(uri).read_bytes()
        f.write(data)
    return dataset

def run_flow(*, flow: object, run_name: str, flow_name: str=None, dataset: str, column_mapping: dict[str,str], run: Optional[str] = None, init_vars: dict = {}) -> pd.DataFrame:
    df = pd.DataFrame()
    if not flow_name:
        try:
            flow_name = flow.__name__.lower()
        except:
            flow_name = str(flow).lower()

    df = exec_pf(flow=flow, run_name=run_name, flow_name=flow_name, dataset=dataset, column_mapping=column_mapping, run=run, init_vars=init_vars)

    return df

def exec_pf(*, flow, run_name, flow_name, dataset, column_mapping, run, init_vars) -> pd.DataFrame:
    logger.info(dict(message=f"Starting {flow_name} with flow {flow} with name: {run_name}"))
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

def run_local(
    *,
    num_runs=1,
    flow,
    target: str,
    model: str,
    batch_id: dict,
    colour: str = "magenta",
    init_vars: dict = {},
    dataset: str|pd.DataFrame,
    column_mapping: dict[str, str]
) -> pd.DataFrame:
    results = []

    if isinstance(dataset, str):
        dataset = pd.read_json(dataset, orient="records", lines=True).sample(frac=1)
    runs = itertools.product(range(num_runs), dataset.iterrows())
    for i, (_, row) in tqdm.tqdm(
        runs,
        total=num_runs*dataset.shape[0],
        colour=colour,
        desc=f"{target}-{model}",
        bar_format="{desc:30}: {bar:20} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        runnable = flow(**init_vars)

        # TODO: convert column_mapping to work for our dataframe
        input_vars = {key: row[mapped] for key, mapped in column_mapping.items() if mapped in row}

        # Run the flow for this line of data
        details = runnable(**input_vars)

        # add input details to  the result row
        for k, v in input_vars.items():
            if k not in details:
                details[k] = v
        details["timestamp"] = pd.to_datetime(datetime.datetime.now())
        for k, v in batch_id.items():
            if k not in details:
                details[k] = v
        for k, v in init_vars.items():
            if k not in details:
                details[k] = v
        row[f"{target}_model"] = model
        row[f"{target}_i"] = i
        row[target] = details

        results.append(row)

    results = pd.DataFrame(results)
    logger.info(
        f"Run for {flow} completed locally, processed {results.shape[0]} results."
    )
    del flow
    torch.cuda.empty_cache()
    gc.collect()
    return results


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    global bm, logger
    bm = BM(cfg=cfg)
    logger = bm.logger

    # from promptflow.tracing import start_trace, trace
    # start_trace(resource_attributes={"run_id": bm._run_id, "job": cfg.project.job}, collection=cfg.project.name)
    # Create a cycle iterator for the progress bar colours
    bar_colours = ["cyan", "yellow", "magenta", "green", "blue"]
    colour_cycle = cycle(bar_colours)

    df = run(data=cfg.data, judger=cfg.judger, evaluator=cfg.evaluator)
    pass

def run(*, data, judger, evaluator):
    global bm
    df = pd.DataFrame()
    data_file = cache_data(data.uri)
    num_runs = judger.get('num_runs',1)
    connections = bm._connections_azure
    # Run judger flow
    try:
        conn = {judger.model: connections[judger.model]}
        init_vars = {"model": judger.model, "standards_path": judger.standard, "template_path": "judge.jinja2", "connection": conn}
        init_vars.update( judger.get("init",{}))
        batch_id = dict(
            run_id=bm._run_id, step=judger.name,
            dataset=data.name,
            model=judger.model, standard=judger.standard.replace(".jinja2", "").replace(".yaml", ""),
        )
        run_name = "_".join(list(batch_id.values()))
        flow_outputs = run_flow(flow=Judger,
                                flow_name=judger.name,
                                dataset=data_file,
                                run_name = run_name,
                                column_mapping=dict(data.columns),
                                init_vars=init_vars)

        # Set up  empty dataframe with batch details ready  to go
        df = pd.json_normalize(itertools.repeat(batch_id, flow_outputs.shape[0]))
        df = pd.concat([df, flow_outputs], axis='columns')

        # Run the evaulation flows
        for eval_model in evaluator.models:
            eval_name = f"{run_name}_evaluator_{eval_model}"
            init_vars = {"model": eval_model}
            flow_name = f'evaluator_{eval_model}'
            evals = run_flow(flow=EvalQA,
                            flow_name=flow_name,
                            dataset=data_file,
                            run = run_name,
                            run_name = eval_name,
                            column_mapping=dict(evaluator.columns),
                            init_vars = init_vars
                    )

            # join the evaluation results
            try:
                df.loc[:, flow_name] = evals[flow_name].to_dict()
                if 'groundtruth' in evals and 'groundtruth' not in df.columns:
                    df.loc[:, 'groundtruth'] = evals['groundtruth']
            except Exception as e:
                # We might not get all the responses back. Try to join on line number instead?
                pass
                df = df.merge(evals[['line_number',flow_name]], left_on='line_number', right_on='line_number')

        # set index
        idx = [x for x in batch_id.keys()]
        df = df.set_index(idx)

    # except Exception as e:
    #     logger.error(f"Unhandled error in our flow: {e}")
    #     raise(e)
    finally:
        if df.shape[0]>0:
            uri = bm.save(df.reset_index(), basename='batch')
            logger.info(
                dict(
                    message=f"Completed batch: {batch_id} with {df.shape[0]} step results saved to {uri}.",
                    **batch_id,results=uri
                )
            )
    return df



if __name__ == "__main__":
    main()
