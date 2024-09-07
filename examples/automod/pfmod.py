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
from buttermilk.flows.evaluate.evaluate import Evaluator
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

def run_flow(*, flow: object, run_name: str, flow_name: Optional[str] = None, dataset: str, column_mapping: dict[str,str], run: Optional[str] = None, init_vars: dict = {}) -> pd.DataFrame:
    pflocal = LocalPFClient()
    df = pd.DataFrame()
    flow_name = flow_name or flow.__name__.lower()
    columns = col_mapping_hydra_to_pf(column_mapping)
    logger.info(dict(message=f"Starting run with flow {flow} with name: {run_name}"))
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
        )

    details = pflocal.get_details(task.name)

    # Stack and rename columns to a predictable format
    inputs = details[[x for x in details.columns if x.startswith('inputs.')]]
    inputs.columns = [x.replace('inputs.', '') for x in inputs.columns]

    id_cols = [x for x in ['record_id', 'line_number'] if x in inputs.columns]
    df = inputs[id_cols]

    # Also a separate column with other step inputs
    df.loc[:, 'inputs'] = inputs.drop(columns=id_cols).to_dict(orient='records')

    # Add a column with the step results
    flow_outputs = details[[x for x in details.columns if x.startswith('outputs.')]]
    flow_outputs.columns = [x.replace('outputs.', '') for x in flow_outputs.columns]
    df.loc[:, flow_name] = flow_outputs.to_dict(orient='records')

    # TODO: see if we can add groundtruth back in
    if 'groundtruth' in flow_outputs and 'groundtruth' not in df.columns:
        df.loc[:, 'groundtruth'] = flow_outputs['groundtruth']

    logger.info(
        f"Run {task.name} for {flow} completed with status {task.status}. URL: {task._portal_url}. Processed {df.shape[0]} results."
    )

    return df

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    pass
    bm = BM(cfg=cfg)
    logger = bm.logger
    results = pd.DataFrame()

    # Create a cycle iterator for the progress bar colours
    bar_colours = ["cyan", "yellow", "magenta", "green", "blue"]
    colour_cycle = cycle(bar_colours)

    data_file = cache_data(cfg.experiments.data.uri)

    connections = bm._connections_azure

    # Judging step
    step = cfg.experiments.judger
    models: list = OmegaConf.to_object(step.get("models", []))
    standards: list = OmegaConf.to_object(step.get("standards", []))
    shuffle(models)
    shuffle(standards)
    permutations = itertools.product(models, standards)
    step_name = step.name
    num_runs = step.get("num_runs", 1)
    for model, standard in permutations:
        df = pd.DataFrame()
        try:
            conn = {model: bm._connections_azure[model]}
            init_vars = {"model": model, "standards_path": standard, "template_path": "judge.jinja2", "connection": conn}
            init_vars.update( step.get("init",{}))
            batch_id = dict(
                run_id=bm._run_id, step="judger",
                dataset=cfg.experiments.data.name,
                model=model, standard=standard.replace(".jinja2", "").replace(".yaml", ""),
            )
            run_name = "_".join(list(batch_id.values()))
            flow_outputs = run_flow(flow=Judger,
                        dataset=data_file,
                        run_name = run_name,
                        column_mapping=dict(cfg.experiments.data.columns), init_vars=init_vars )

            # Set up  empty dataframe with batch details ready  to go
            df = pd.json_normalize(itertools.repeat(batch_id, flow_outputs.shape[0]))
            df = pd.concat([df, flow_outputs], axis='columns')

            # Run the evaulation flows
            for eval_model in cfg.experiments.evaluator.models:
                eval_name = f"{run_name}_evaluator_{eval_model}"
                init_vars = {"model": eval_model}
                flow_name = f'evaluator_{model}'
                evals = run_flow(flow="buttermilk/flows/evaluate",
                                 flow_name=flow_name,
                                dataset=data_file,
                                run = run_name,
                                run_name = eval_name,
                                column_mapping=dict(cfg.experiments.evaluator.columns),
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

        except Exception as e:
            logger.error(f"Unhandled error in our flow: {e}")
            continue
        finally:
            if df.shape[0]>0:
                uri = bm.save(df.reset_index())
                logger.info(
                    dict(
                        message=f"Completed batch: {batch_id} with {df.shape[0]} step results saved to {uri}.",
                        **batch_id,results=uri
                    )
                )
                # add to full batch results
                results = pd.concat([results, df])

    try:
        # generate metrics
        metriciser = Metriciser()
        metrics = metriciser.evaluate_results(df, col=step_name)
        metrics_uri = bm.save(metrics.reset_index(), basename='metrics.jsonl')
        logger.info(dict(message=f"Full run completed, saved {metrics.shape[0]} aggregated metrics to {metrics_uri}", **batch_id,results=uri
            ))
    except Exception as e:
        logger.error(f"Unhandled error generating metrics: {e}")

    finally:
        if results.shape[0]>0:
            uri = bm.save(results.reset_index(), basename="results")
            logger.info(dict(message=f"Full run completed, saved {results.shape[0]} final batch results to {uri}", **batch_id,results=uri
                ))
        else:
            logger.info(dict(message=f"Full run completed, no results to save.", **batch_id
                ))


    pass


if __name__ == "__main__":
    run()
