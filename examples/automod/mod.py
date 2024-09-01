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
from buttermilk.flows.results_bq import SaveResultsBQ
from promptflow.azure import PFClient as AzurePFClient
from promptflow.client import PFClient as LocalPFClient
import hydra
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


def run_local(
    *,
    target: str,
    model: str,
    batch_id: dict,
    colour: str = "magenta",
    dataset: str,
    column_mapping: dict[str, str],
    init: dict = {},
) -> pd.DataFrame:
    results = []
    init = init or {}

    if target == "moderate":
        flow = Moderator(model=model, **init)
    else:
        flow = Analyst(model=model, **init)
    df = pd.read_json(dataset, orient="records", lines=True).sample(frac=1)
    for _, row in tqdm.tqdm(
        df.iterrows(),
        colour=colour,
        desc=f"{target}-{model}",
        bar_format="{desc:30}: {bar:20} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        input_vars = {key: row[mapped] for key, mapped in column_mapping.items()}
        details = flow(**input_vars)
        details["init_vars"] = init
        details["record"] = row.to_dict()
        details["timestamp"] = pd.to_datetime(datetime.datetime.now())
        for k, v in batch_id.items():
            if k not in details:
                details[k] = v
        results.append(details)

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

    for step, step_config in cfg.experiments.items():
        data_file = cache_data(cfg.data.uri)
        logger.info(
            f"Cached local copy of dataset {cfg.data.name} from {cfg.data.uri} to {data_file}"
        )

        logger.debug(f"Running {step} {step_config.name}")
        models: list = OmegaConf.to_object(step_config.get("models", []))
        shuffle(models)
        for model in models:
            try:
                batch_id = dict(
                    run_id=bm._run_id,
                    experiment=step_config.name,
                    dataset=cfg.data.name,
                    model=model,
                )

                logger.info(f"Running batch: {batch_id}")
                df = run_local(
                    model=model,
                    target=step,
                    dataset=data_file,
                    batch_id=batch_id,
                    column_mapping=cfg.data.columns,
                    colour=next(colour_cycle),
                    init=step_config.get("init", {}),
                )
                logger.info(
                    dict(
                        message=f"Successfully  completed batch: {batch_id} with {df.shape[0]} results.",
                        **batch_id,
                    )
                )
            except Exception as e:
                logger.error(f"Unhandled error in our flow: {e}")
                break
            finally:
                uri = bm.save(df.reset_index())
                logger.info(f"Saved {df.shape[0]} step results to {uri}")
                results = pd.concat([results, df])
        Path(data_file).unlink(missing_ok=True)

    uri = bm.save(results.reset_index(), filename="results")
    logger.info(f"Saved {results.shape[0]} run batch results to {uri}")


if __name__ == "__main__":
    run()
