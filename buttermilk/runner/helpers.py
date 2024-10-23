import json
from tempfile import NamedTemporaryFile

import cloudpathlib
import pandas as pd

from buttermilk import BM, logger
from buttermilk.utils.flows import col_mapping_hydra_to_local
from typing import Mapping, Optional, Sequence

from buttermilk.utils.utils import find_key_string_pairs

def load_data(data_cfg) -> pd.DataFrame:
    if data_cfg.type == 'file':
        df = pd.read_json(data_cfg.uri, lines=True, orient='records')
        # convert column_mapping to work for our dataframe
        columns = col_mapping_hydra_to_local(data_cfg.columns)
        rename_dict = {v: k for k, v in columns.items()}
        df = df.rename(columns=rename_dict)
    elif data_cfg.type == 'job':
        df = load_jobs(dataset=data_cfg.dataset, filter=data_cfg.filter, last_n_days=data_cfg.last_n_days, exclude_processed=data_cfg.group)

    return df

def load_jobs(dataset: str, filter: dict = {}, last_n_days=3, exclude_processed: list=[]) -> pd.DataFrame:
    sql = f"SELECT * FROM `{dataset}` jobs WHERE error IS NULL "

    sql += f" AND TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL {last_n_days} DAY)) "

    for field, condition in filter.items():
        if condition and isinstance(condition, str):
            sql += f" AND {field} = '{condition}' "
        elif condition and isinstance(condition, Sequence):
            multi_or = " OR ".join([f"{field} = '{term}'" for term in condition])
            sql += f" AND ({multi_or})"

    sql += " AND NOT prediction IS NULL "

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


def group_and_filter_jobs(new_data: pd.DataFrame, group: dict, columns: dict, df: Optional[pd.DataFrame] = None, max_records_per_group: int = -1) -> pd.DataFrame:

    # expand and rename columns if we need to
    pairs_to_expand = list(find_key_string_pairs(group)) + list(find_key_string_pairs(columns))
    
    for col_name, grp in pairs_to_expand:
        try:
            grp, col = grp.split('.', 1)

            # extract the contents of the nested source column that
            # will form the new index

            # First, check if the column is a JSON string, and interpret it.
            try:
                new_data[grp] = new_data[grp].apply(json.loads)
            except TypeError:
                pass  # already a dict
            try:
                # Now, try to get the sub-column from the dictionary within the grp column
                new_data.loc[:, col_name] = pd.json_normalize(new_data[grp])[col].values
            except Exception as e:
                logger.exception(f"Error extracting column {col} from {grp} in {new_data}: {e} {e.args=}")
                raise e

        except ValueError:
            pass  # no group in this column definition
            if col_name != grp:
                # rename column
                new_data = new_data.rename(columns={grp: col_name})

    # Add columns to group by to the index
    idx_cols = [ c for c in group.keys() if c in new_data.columns]

    # Stack any nested fields in the mapping
    for k, v in columns.items():
        if isinstance(v, Mapping):
            # put all the mapped columns into a dictionary in
            # a new field named as provided in the step config
            new_data.loc[:, k] = new_data[v.keys()].to_dict(orient='records')

    if max_records_per_group > 0:
        # Reduce down to n list items per index (but don't aggregate
        # at this time, just keep a random selection of rows)
        new_data = new_data.sample(frac=1).groupby(idx_cols).agg(
                lambda x: x.tolist()[:max_records_per_group])



    # Add the column to the source dataset
    if df is not None and df.shape[0]>0:
        # reset index columns that we're not matching on:
        group_only_cols = [x for x in idx_cols if x not in df.columns]
        idx_cols = list(set(idx_cols).difference(group_only_cols))
        new_data = new_data.reset_index(level=group_only_cols, drop=False)

        # Only return the columns we need
        new_data = new_data[columns.keys()]

        df = pd.merge(df, new_data, left_on=idx_cols, right_index=True)
        return df
    else:
        # Only return the columns we need
        return new_data[idx_cols + list(columns.keys())]





def cache_data(uri: str) -> str:
    with NamedTemporaryFile(delete=False, suffix=".jsonl", mode="wb") as f:
        dataset = f.name
        data = cloudpathlib.CloudPath(uri).read_bytes()
        f.write(data)
    return dataset