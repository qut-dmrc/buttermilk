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
from buttermilk.flows.evaluate.score import Evaluator
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
from typing import Type, TypeVar, Callable

logger = getLogger()

def run_flow(*, flow: Callable, dataset: str, column_mapping: dict[str,str]) -> pd.DataFrame:
    pflocal = LocalPFClient()

    standards = read_text(standards_path)
    init_vars = dict(model = model, criteria=standards, template=template_path)

    flow = Analyst(**init_vars)

    run = pflocal.run(
            flow=flow,
            data=dataset,
            init_vars=init_vars,
            column_mapping=column_mapping,
            stream=False,
            name=batch_name,
            timeout=150,
        )

    logger.info(
        f"Run {run.name} completed with status {run.status}. URL: {run._portal_url}."
    )

    details = pflocal.get_details(run.name)
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

    dataset = pd.read_json(cfg.experiments.data.uri, orient='records', lines=True)

    # Judging step
    if step := cfg.experiments.get('judge'):
        models: list = OmegaConf.to_object(step.get("models", []))
        standards: list = OmegaConf.to_object(step.get("standards", []))
        shuffle(models)
        shuffle(standards)
        permutations = itertools.product(models, standards)
        step_name = step.name
        num_runs = step.get("num_runs", 1)
        for model, standard in permutations:

            df = pd.DataFrame()

            judge = Judger(model=model, standards_path=standard, **step.get("init",{}))
            metriciser = Metriciser()

            try:
                batch_id = dict(
                    run_id=bm._run_id, step="judge",
                    dataset=cfg.experiments.data.name,
                    model=model, standard=standard,
                )

                logger.info(dict(message=f"Judging {dataset.shape[0]} records over {num_runs} iteration(s) for batch: {batch_id}", **batch_id))
                df = run_local(
                    flow=judge,
                    model=model,
                    num_runs=num_runs,
                    target=step_name,
                    dataset=dataset,
                    batch_id=batch_id,
                    column_mapping=cfg.experiments.data.columns,
                    colour=next(colour_cycle)
                )
                uri = bm.save(df.reset_index())
                logger.info(
                    dict(
                        message=f"Successfully completed batch: {batch_id} with {df.shape[0]} step results saved to {uri}.",
                        **batch_id,results=uri
                    )
                )

                # evaluate
                for model in cfg.experiments.evals.get("evaluator", {}).get("models",{}):
                    evaluator = Evaluator(model=model, **cfg.experiments.evals.get("evaluator", {}).get("init",{}))
                    evals = evaluator.batch(df, prediction=step_name)
                    uri = bm.save(evals, basename=f'evaluator_{model}.jsonl')
                    logger.info(dict(message=f"Saved {df.shape[0]} evaluated results from {model} to {uri}", **batch_id,results=uri
                        ))
                    df.loc[:, f'evaluator_{model}'] = evals

                # Add all identifying details from batch_id and set index
                df = pd.concat([df, pd.json_normalize(itertools.repeat(batch_id, df.shape[0]))], axis='columns')
                idx = [x for x in batch_id.keys()]
                df = df.set_index(idx)

                # generate metrics
                metrics = metriciser.evaluate_results(df, col=step_name, levels=idx)
                uri = bm.save(metrics.reset_index(), basename='metrics.jsonl')
                logger.info(dict(message=f"Saved {metrics.shape[0]} aggregated scored batch results to {uri}", **batch_id,results=uri
                    ))

            except Exception as e:
                logger.error(f"Unhandled error in our flow: {e}")
                continue
            finally:
                results = pd.concat([results, df])
            pass


    uri = bm.save(results.reset_index(), basename="results")
    logger.info(f"Saved {results.shape[0]} run results to {uri}")


if __name__ == "__main__":
    run()
