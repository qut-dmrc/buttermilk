import datetime
import resource
from multiprocessing import Process
from pathlib import Path
from random import shuffle
from tempfile import NamedTemporaryFile
import gc
import cloudpathlib
import pandas as pd
from sqlalchemy import column
from buttermilk import BM
from buttermilk.flows.extract import Analyst
from buttermilk.flows.moderate.scorers import Moderator
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

def cache_data(uri: str) -> str:
    with NamedTemporaryFile(delete=False, suffix=".jsonl", mode="wb") as f:
        dataset = f.name
        data = cloudpathlib.CloudPath(uri).read_bytes()
        f.write(data)
    return dataset

def run_flow(*, flow: object, run_name: str, dataset: str, column_mapping: dict[str,str], run: Optional[str] = None, init_vars: dict = {}) -> pd.DataFrame:
    pflocal = LocalPFClient()

    logger.info(dict(message=f"Starting run with flow {flow} with name: {run_name}"))
    task = pflocal.run(
            flow=flow,
            data=dataset,
            run=run,
            init_vars=init_vars,
            column_mapping=column_mapping,
            stream=False,
            name=run_name,
            timeout=150,
        )

    details = pflocal.get_details(task.name)

    logger.info(
        f"Run {task.name} for {flow} completed with status {task.status}. URL: {task._portal_url}. Processed {details.shape[0]} results."
    )

    return details

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
    step = cfg.experiments.judge
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
            init_vars = {"model": model, "standards_path": standards, "template_path": "judge.jinja2", "connection": conn}
            init_vars.update( step.get("init",{}))
            batch_id = dict(
                run_id=bm._run_id, step="judge",
                dataset=cfg.experiments.data.name,
                model=model, standard=standard,
            )
            run_name = f"{bm._run_id}_{step.name}_{cfg.experiments.data.name}_{model}_{standard}"

            flow_outputs = run_flow(flow=Judger,
                        dataset=data_file,
                        run_name = run_name,
                        column_mapping=OmegaConf.to_object(cfg.experiments.data.columns) )

            # Set up  empty dataframe with batch details ready  to go
            df = pd.json_normalize(itertools.repeat(batch_id, flow_outputs.shape[0]))

            # Add a column with the step results
            df.loc[:, step.name] = flow_outputs.to_dict(orient='records')

            # set index
            idx = [x for x in batch_id.keys()]
            df = df.set_index(idx)

            # Run the evaulation flows
            for eval_model in cfg.experiments.evaluator.models:
                eval_name = f"{run_name}_evaluator_{eval_model}"
                init_vars = {"model": eval_model}
                evals = run_flow(flow=Evaluator,
                            dataset=data_file,
                            run = run_name,
                            run_name = eval_name,
                            column_mapping=OmegaConf.to_object(cfg.experiments.evaluator.columns))

                # Add a column with the evaluation results
                df.loc[:, f'evaluator_{model}'] = evals.to_dict(orient='records')

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
