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
from buttermilk.apis import Llama2ChatMod, replicatellama2, replicatellama3, HFInferenceClient, HFLlama270bn, HFPipeline
BASE_DIR = Path(__file__).absolute().parent
import torch
import datetime
import tqdm
from itertools import cycle
from buttermilk.toxicity  import *

import cloudpathlib
from tempfile import NamedTemporaryFile

def run_local(*, target: str, model: str, colour:str='magenta', dataset: str, column_mapping: dict[str,str], init: dict, batch_id: dict) -> pd.DataFrame:
    results = []
    run_meta = {"timestamp": pd.to_datetime(datetime.datetime.now())}
    run_meta.update(batch_id)

    if target == 'moderate':
        flow = Moderator(**init)
    else:
        flow = Analyst(**init)
    df = pd.read_json(dataset, orient='records', lines=True).sample(frac=1)
    for _, row in tqdm.tqdm(df.iterrows(),colour=colour):
        input_vars = {key: row[mapped] for key, mapped in column_mapping.items()}
        details = flow(**input_vars)
        details['run_meta'] = run_meta.copy()
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


    data_file = cache_data(cfg.dataset.uri)
    logger.info(f"Cached local copy of dataset {cfg.dataset.name} from {cfg.run.dataset.uri} to {data_file}")

    # Create a cycle iterator for the progress bar colours
    bar_colours =  ["cyan", "yellow","magent","green", "blue"]
    colour_cycle = cycle(bar_colours)

    for step, step_config in cfg.experiments.keys():
        logger.debug(f"Running {step} {step_config.name}")
        models: list = OmegaConf.to_object(step_config.get('models', []))
        shuffle(models)
        for model in models:
            try:
                batch_id = dict(run_id=bm._run_id,
                                experiment=step_config.name,
                                model=model,
                                dataset=cfg.dataset.name)

                logger.info(f"Running batch: {batch_id}")
                df = run_local(model=model, target=step,
                                dataset=data_file,
                                column_mapping=cfg.dataset.columns,
                                batch_id=batch_id,
                                colour=next(colour_cycle),
                                init=step_config.init)
                logger.info(f"Successfully  completed batch: {batch_id}  with {df.shape[0]} results.")
                results = pd.concat([results, df])
            except Exception as e:
                logger.error(f"Unhandled error in our flow: {e}")
                break

    bm.save(results.reset_index())
    Path(data_file).unlink(missing_ok=True)


if __name__ == "__main__":
    run()