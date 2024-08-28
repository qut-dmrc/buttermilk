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

import datetime
import cloudpathlib
from tempfile import NamedTemporaryFile

BASE_DIR = Path(__file__).absolute().parent

def run_batch(*, model: str, dataset: str, column_mapping: dict[str,str], standards_path: str, template_path: str, batch_name: str) -> pd.DataFrame:
    bm = BM()
    logger = bm.logger
    results = pd.DataFrame()
    pflocal = LocalPFClient()

    init_vars = dict(model = model)

    flow = Moderator(**init_vars)

    run_meta = {"name": batch_name, "model": model, "timestamp": pd.to_datetime(datetime.datetime.now())}
    run = pflocal.run(
            flow=flow.moderate,
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

if __name__ == "__main__":
    run()