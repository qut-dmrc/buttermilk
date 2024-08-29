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
from buttermilk.flows.moderate.scorers import Moderator
from buttermilk.flows.results_bq import SaveResultsBQ
from promptflow.azure import PFClient as AzurePFClient
from promptflow.client import PFClient as LocalPFClient
import hydra
from buttermilk.utils.utils import read_text
from omegaconf import DictConfig, OmegaConf
from buttermilk.apis import Llama2ChatMod, replicatellama2, replicatellama3, HFInferenceClient, HFLlama270bn, HFPipeline
BASE_DIR = Path(__file__).absolute().parent

import datetime


import cloudpathlib
from tempfile import NamedTemporaryFile

def run_ots(*, logger, model: str, dataset: str, column_mapping: dict[str,str], batch_name: str) -> pd.DataFrame:

    from buttermilk.toxicity.toxicity  import TOXCLIENTS, TOXCLIENTS_LOCAL
    pflocal = LocalPFClient()
    init_vars = dict(model = model)

    flow = Moderator(**init_vars)

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

def run_flow(*, logger, model: str, dataset: str, column_mapping: dict[str,str], standards_path: str, template_path: str, batch_name: str) -> pd.DataFrame:
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

def run_local(*, model: str, model_type, dataset: str, column_mapping: dict[str,str], standards_path: str, template_path: str, batch_name: str) -> pd.DataFrame:
    bm = BM()
    logger = bm.logger
    results = []
    run_meta = {"name": batch_name, "model_type": model_type, "model": model, "timestamp": pd.to_datetime(datetime.datetime.now())}

    init_vars = dict(model = model)
    if model_type == 'ots':
        flow = Moderator(**init_vars)
    else:
        flow = Analyst(**init_vars)

    for _, row in pd.read_json(dataset, orient='records', lines=True).sample(frac=1).iterrows():
        input_vars = {key: row[mapped] for key, mapped in bm.cfg.run.dataset.columns.items()}
        details = flow(**input_vars)
        details['run_meta'] = run_meta.copy()
        results.append(details)

    logger.info(
        f"Run {run.name} completed with status {run.status}. URL: {run._portal_url}."
    )
    results = pd.DataFrame(results)
    return results

def run_batch(*, model: str, model_type, dataset: str, column_mapping: dict[str,str], standards_path: str, template_path: str, batch_name: str) -> pd.DataFrame:
    bm = BM()
    logger = bm.logger
    results = pd.DataFrame()
    run_meta = {"name": batch_name, "model_type": model_type, "model": model, "timestamp": pd.to_datetime(datetime.datetime.now())}

    if model_type == 'ots':
        details = run_ots(logger=logger, model=model, dataset=dataset, column_mapping=column_mapping, batch_name=batch_name)
    elif model_type == 'flow':
        details = run_flow(logger=logger, model=model, dataset=dataset, column_mapping=column_mapping, standards_path=standards_path, template_path=template_path, batch_name=batch_name)

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
    bm = BM(cfg=cfg)
    logger = bm.logger
    results = pd.DataFrame()

    dataset = cache_data(cfg.run.dataset.uri)
    logger.info(f"Caching local copy of dataset from {cfg.run.dataset.uri} to {dataset}")

    models = OmegaConf.to_object(cfg.run.tasks.judge.models)
    shuffle(list(models))
    for model in models:
        try:
            batch_name = f"{bm._run_id}_{model}_{cfg.run.tasks.judge.experiment_name}"
            columns = OmegaConf.to_object(cfg.run.dataset.columns)
            model_type = cfg.run.tasks.judge.model_type
            logger.info(f"Running {batch_name} with {model_type} on {cfg.run.get('style', 'pf batch')}")
            if cfg.run.get("style") == 'local':
                df = run_local(model=model,model_type=model_type,
                            dataset=dataset,
                            column_mapping=columns,
                            standards_path=cfg.run.tasks.judge.standards,
                            template_path=cfg.run.tasks.judge.template,
                            batch_name=batch_name)
            else:
                df = run_batch(model=model,model_type=model_type,
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