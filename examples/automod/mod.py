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

def run_local(
    *,
    num_runs=1,
    flow,
    target: str,
    model: str,
    batch_id: dict,
    colour: str = "magenta",
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
        input_vars = {key: row[mapped] for key, mapped in column_mapping.items() if mapped in row}
        details = flow(**input_vars)
        details["timestamp"] = pd.to_datetime(datetime.datetime.now())
        for k, v in batch_id.items():
            if k not in details:
                details[k] = v
        row[f"{target}_model"] = model
        row[f"{target}_i"] = i
        row[target] = details

        results.append(row)
    results = pd.DataFrame(results)
    del flow
    torch.cuda.empty_cache()
    gc.collect()
    return results


def cache_data(uri: str) -> str:
    with NamedTemporaryFile(delete=False, suffix=".jsonl", mode="wb") as f:
        dataset = f.name
        data = cloudpathlib.CloudPath(uri).read_bytes()
        f.write(data)
    return dataset


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
