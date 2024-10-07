from tempfile import NamedTemporaryFile

import cloudpathlib
import pandas as pd

from buttermilk import BM
from buttermilk.utils.flows import col_mapping_hydra_to_local


def load_data(data_cfg) -> pd.DataFrame:
    if data_cfg.type == 'file':
        df = pd.read_json(data_cfg.uri, lines=True, orient='records')
    elif data_cfg.type == 'job':
        df = load_job(dataset=data_cfg.dataset, filter=data_cfg.filter)
    
    # convert column_mapping to work for our dataframe
    columns = col_mapping_hydra_to_local(data_cfg.columns)
    rename_dict = {v: k for k, v in columns.items()}

    df = df.rename(columns=rename_dict)

    return df

def load_job(dataset: str, filter: dict = {}) -> pd.DataFrame:
    sql = f"SELECT * FROM `{dataset}` WHERE error IS NULL "
    
    for field, condition in filter.items():
        sql += f" AND {field} = '{condition}' "

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