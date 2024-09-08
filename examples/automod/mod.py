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

    dataset = pd.read_json(cfg.experiments.data.uri, orient='records', lines=True)

        judge = Judger(model=model, standards_path=standard, **step.get("init",{}))

            df = run_local(
                flow=judge,
                model=model,
                num_runs=num_runs,
                target=step_name,
                dataset=dataset,
                batch_id=batch_id,
                column_mapping=OmegaConf.to_object(cfg.experiments.data.columns),
                colour=next(colour_cycle)
            )

        evals = evaluator.batch(df, prediction=step_name)


if __name__ == "__main__":
    run()
