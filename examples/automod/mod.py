import datetime
import resource
from multiprocessing import Process
from pathlib import Path
from random import shuffle
from tempfile import NamedTemporaryFile

import cloudpathlib
import pandas as pd
from sqlalchemy import column
from buttermilk import BM
from buttermilk.flows.extract import Analyst
from buttermilk.flows.results_bq import SaveResultsBQ
from promptflow.azure import PFClient as AzurePFClient
from promptflow.client import PFClient as LocalPFClient
import hydra
from buttermilk.utils.utils import read_text
from omegaconf import DictConfig, OmegaConf

BASE_DIR = Path(__file__).absolute().parent

import datetime


import cloudpathlib
from tempfile import NamedTemporaryFile

def run_batch(*, model: str, dataset: str, column_mapping: dict[str,str], standards_path: str, template_path: str, batch_name: str) -> pd.DataFrame:
    bm = BM()
    logger = bm.logger
    results = pd.DataFrame()
    pflocal = LocalPFClient()

    standards = read_text(standards_path)
    init_vars = dict(model = model, criteria=standards, template=template_path)

    flow = Analyst(**init_vars)

    run_meta = {"name": batch_name, "model": model, "timestamp": pd.to_datetime(datetime.datetime.now())}
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

    # duplicate run_info metadata for each row
    run_meta = pd.DataFrame.from_records([run_meta for _ in range(details.shape[0])])
    details = pd.concat([details, run_meta], axis='columns')

    # Add to any other results we have
    results = pd.concat([results, details], axis='index')

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
    bm = BM()
    logger = bm.logger
    results = pd.DataFrame()

    #Set to Fork instead of Spawn
    import os
    os.environ['PF_BATCH_METHOD']='fork'

    dataset = cache_data(cfg.run.dataset.uri)
    logger.info(f"Caching local copy of dataset from {cfg.run.dataset.uri} to {dataset}")

    models = OmegaConf.to_object(cfg.run.tasks.judge.models)
    shuffle(list(models))
    for model in cfg.run.tasks.judge.models:
        try:
            batch_name = f"{bm._run_id}_{model}_{cfg.run.tasks.judge.experiment_name}"
            columns = OmegaConf.to_object(cfg.run.dataset.columns)
            df = run_batch(model=model,
                           dataset=dataset,
                           column_mapping=columns,
                           standards_path=cfg.run.tasks.judge.standards,
                           template_path=cfg.run.tasks.judge.template,
                           batch_name=batch_name)
            results = pd.concat([results, df])
        except Exception as e:
            logger.error(f"Unhandled error in our flow: {e}")
            break

    bm.save(results)
    Path(dataset).unlink(missing_ok=True)


if __name__ == "__main__":
    run()