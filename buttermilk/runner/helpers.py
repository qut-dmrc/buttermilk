from tempfile import NamedTemporaryFile

import cloudpathlib
import pandas as pd

from buttermilk import BM
from buttermilk.utils.flows import col_mapping_hydra_to_local


def load_data(data_cfg, new_job_name='') -> pd.DataFrame:
    if data_cfg.type == 'file':
        df = pd.read_json(data_cfg.uri, lines=True, orient='records')
    elif data_cfg.type == 'job':
        df = load_job(dataset=data_cfg.dataset, filter=data_cfg.filter, last_n_days=data_cfg.last_n_days, exclude_processed=data_cfg.group, new_job_name=new_job_name)

    # convert column_mapping to work for our dataframe
    columns = col_mapping_hydra_to_local(data_cfg.columns)
    rename_dict = {v: k for k, v in columns.items()}

    df = df.rename(columns=rename_dict)

    return df

def load_job(dataset: str, filter: dict = {}, last_n_days=3, exclude_processed: list=[], new_job_name: str='' ) -> pd.DataFrame:
    sql = f"SELECT * FROM `{dataset}` jobs WHERE error IS NULL "

    sql += f" AND TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL {last_n_days} DAY)) "

    for field, condition in filter.items():
        if condition:
            sql += f" AND {field} = '{condition}' "

    sql += " AND NOT expected IS NULL "

    # exclude records already processed if necessary
    # if exclude_processed:
    #     sql += f" AND job_id NOT IN (SELECT job_id FROM `{dataset}` processed WHERE error IS NULL AND "
    #     sql += f"processed.step = '{new_job_name}' AND "
    #     for level in exclude_processed:
    #         sql += f"processed.{level} = jobs.{level} AND "
    #     sql += f" TIMESTAMP_TRUNC(processed.timestamp, DAY) >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL {last_n_days} DAY)) "
    #     sql += ") "

    sql += " ORDER BY RAND() "

    bm = BM()
    df = bm.run_query(sql)

    return df


def cache_data(uri: str) -> str:
    with NamedTemporaryFile(delete=False, suffix=".jsonl", mode="wb") as f:
        dataset = f.name
        data = cloudpathlib.CloudPath(uri).read_bytes()
        f.write(data)
    return dataset