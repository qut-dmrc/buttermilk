import json
from collections.abc import Mapping, Sequence
from tempfile import NamedTemporaryFile

import cloudpathlib
import pandas as pd

from buttermilk import BM, logger
from buttermilk._core.config import DataSource
from buttermilk.utils.flows import col_mapping_hydra_to_local
from buttermilk.utils.utils import find_key_string_pairs


async def load_data(data_cfg: DataSource) -> pd.DataFrame:
    if data_cfg.type == "file":
        logger.debug(f"Reading data from {data_cfg.path}")
        df = pd.read_json(data_cfg.path, lines=True, orient="records")
        # convert column_mapping to work for our dataframe
        columns = col_mapping_hydra_to_local(data_cfg.columns)
        rename_dict = {v: k for k, v in columns.items()}
        df = df.rename(columns=rename_dict)

    elif data_cfg.type == "job":
        df = load_jobs(data_cfg=data_cfg)
    elif data_cfg.type == "bq":
        df = load_bq(data_cfg=data_cfg)
    elif data_cfg.type == "plaintext":
        # Load all files in a directory
        df = await read_all_files(
            data_cfg.path,
            data_cfg.glob,
            columns=data_cfg.columns,
        )
    else:
        raise ValueError(f"Unknown data type: {data_cfg.type}")

    return df


def load_bq(data_cfg: DataSource) -> pd.DataFrame:
    sql = f"SELECT * FROM `{data_cfg.path}`"
    sql += " ORDER BY RAND() "

    bm = BM()

    df = bm.run_query(sql)

    return df


def load_jobs(data_cfg: DataSource) -> pd.DataFrame:
    last_n_days = data_cfg.last_n_days

    sql = f"SELECT * FROM `{data_cfg.path}` jobs WHERE error IS NULL AND (JSON_VALUE(outputs, '$.error') IS NULL) "

    sql += f" AND TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL {last_n_days} DAY)) "
    if data_cfg.filter:
        for field, condition in data_cfg.filter.items():
            field = field.split(".", 1)
            if len(field) > 1:
                comparison_string = f'JSON_VALUE({field[0]}, "$.{field[1]}")'
                comparison_string_object = f'JSON_QUERY({field[0]}, "$.{field[1]}")'
            else:
                comparison_string = field[0]
            if condition == [] or condition == ["NOTNULL"]:
                # Where we're testing a JSON array or record, we need to use JSON_QUERY instead of JSON_VALUE (for strings)
                sql += f" AND {comparison_string_object} IS NOT NULL "
            elif not condition or condition == "NOTNULL":
                sql += f" AND {comparison_string} IS NOT NULL "
            elif condition and isinstance(condition, str):
                sql += f' AND {comparison_string} = "{condition}" '
            elif condition and isinstance(condition, Sequence):
                multi_or = " OR ".join([
                    f'{comparison_string} = "{term}"' for term in condition
                ])
                sql += f" AND ({multi_or})"

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


def group_and_filter_jobs(
    *,
    data: pd.DataFrame,
    data_cfg: DataSource,
    existing_dfs: pd.DataFrame | None = None,
    raise_on_error=True,
) -> pd.DataFrame:
    # expand and rename columns if we need to
    pairs_to_expand = (
        list(find_key_string_pairs(data_cfg.join))
        + list(find_key_string_pairs(data_cfg.group))
        + list(find_key_string_pairs(data_cfg.columns))
    )

    for col_name, grp in pairs_to_expand:
        try:
            grp, col = grp.split(".", 1)

            # extract the contents of the nested source column that
            # will form the new index

            # First, check if the column is a JSON string, and interpret it.
            try:
                data[grp] = data[grp].apply(
                    lambda x: json.loads(x) if pd.notna(x) else dict(),
                )
            except TypeError:
                pass  # already a dict
            try:
                # Now, try to get the sub-column from the dictionary within the grp column
                data.loc[:, col_name] = pd.json_normalize(data[grp])[col].values
            except Exception as e:
                logger.error(f"Error extracting column {col} from {grp}: {e} {e.args=}")
                if raise_on_error:
                    raise e

        except ValueError:
            # no group in this column definition
            if col_name != grp:
                # rename column
                data = data.rename(columns={grp: col_name})
        except KeyError as e:
            logger.error(f"Error extracting column {col} from {grp}: {e} {e.args=}")
            if raise_on_error:
                raise e

    # Add columns to group by to the index
    idx_cols = list(data_cfg.join.keys()) + list(data_cfg.group.keys())
    idx_cols = [c for c in idx_cols if c in data.columns]
    data = data.set_index(idx_cols, drop=False)

    # Stack any nested fields in the mapping
    if data_cfg.columns:
        for k, v in data_cfg.columns.items():
            if isinstance(v, Mapping):
                # put all the mapped columns into a dictionary in
                # a new field named as provided in the step config
                data.loc[:, k] = data[v.keys()].to_dict(orient="records")

    if data_cfg.max_records_per_group > 0 and idx_cols:
        # Reduce down to n list items per index (but don't aggregate
        # at this time, just keep a random selection of rows)
        data = (
            data.sample(frac=1)
            .groupby(level=idx_cols, as_index=True)
            .agg(lambda x: x.tolist()[: data_cfg.max_records_per_group])
        )

    # Only return the columns we need
    if data_cfg.columns and len(data_cfg.columns) > 0:
        cols = [x for x in list(data_cfg.columns.keys()) if x in data.columns]
        missing_cols = set(data_cfg.columns.keys()) - set(cols)
        if missing_cols:
            logger.error(f"Missing columns {list(missing_cols)}")
            if raise_on_error:
                raise KeyError(f"Missing columns {list(missing_cols)}")
            for c in missing_cols:
                data[c] = None
        data = data[list(data_cfg.columns.keys())]

    # Join the columns to the existing dataset
    if existing_dfs is not None and existing_dfs.shape[0] > 0:
        if idx_cols:
            # reset index columns that we're not matching on:
            data = data.reset_index(level=list(data_cfg.group.keys()), drop=True)

            # If agg==True, reduce the groups down again now that we have built
            # our index. This way, we should only have one matching row per
            # group for each of the records we're processing.
            if data_cfg.agg:
                data = data.groupby(level=list(data_cfg.join.keys())).agg(
                    lambda x: [item for sublist in x for item in sublist],
                )

        existing_dfs = pd.merge(
            existing_dfs,
            data,
            left_on=list(data_cfg.join.keys()),
            right_index=True,
        )
        return existing_dfs
    return data


def cache_data(uri: str) -> str:
    with NamedTemporaryFile(delete=False, suffix=".jsonl", mode="wb") as f:
        dataset = f.name
        data = cloudpathlib.CloudPath(uri).read_bytes()
        f.write(data)
    return dataset


async def prepare_step_df(data_configs: list[DataSource]) -> dict[str, pd.DataFrame]:
    # This works for small datasets that we can easily read and load.
    datasets = {}
    source_list = []
    dataset_configs = []

    # data_cfg is not ordered. Loop through and load the static data first.
    for src in data_configs:
        if src.type == "job":
            # end of list (dynamic data)
            dataset_configs.append(src)
        else:
            # start of list
            dataset_configs = [src] + dataset_configs
        source_list.append(src.name)

    combined_df = pd.DataFrame()
    for src in dataset_configs:
        new_df = await load_data(src)
        if src.type == "job":
            # Load and join prior job data
            new_df = group_and_filter_jobs(
                existing_dfs=combined_df,
                data=new_df,
                data_cfg=src,
            )
        else:
            # TODO @nicsuzor: group and limit by max_records_per_group
            pass

        if src.columns:
            new_df = new_df[src.columns.keys()]

        # shuffle
        new_df = new_df.sample(frac=1)

        # Also hold the combined data for next iteration
        combined_df = new_df.copy()

        datasets[src.name] = new_df
    return datasets


async def read_all_files(uri, pattern, columns: dict[str, str]):
    filelist = cloudpathlib.GSPath(uri).glob(pattern)
    # Read each file into a DataFrame and store in a list
    dataset = pd.DataFrame(columns=columns.keys())
    for file in filelist:
        logger.debug(f"Reading {file.name} from {file.parent}...")
        content = file.read_bytes().decode("utf-8")
        dataset.loc[len(dataset)] = (file.stem, content)
    return dataset


# async def read_all_files(uri: str, pattern: str, columns: Dict[str, str]) -> pd.DataFrame:
#     filelist = GSPath(uri).glob(pattern)

#     # for testing only
#     filelist = [x for i, x in enumerate(filelist) if i <= 20]

#     fs = gcsfs.GCSFileSystem(asynchronous=True)

#     _sem = asyncio.Semaphore(4)
#     async def read_file(file: cloudpathlib.CloudPath) -> Tuple[str, str]:
#         async with _sem:
#             logger.debug(f"Reading {file.name} from {file.parent}...")
#             content = await fs.read_text(file.as_uri(), encoding='utf-8')
#             return file.stem, content

#     # Create a list of tasks for reading files concurrently
#     tasks = [read_file(file) for file in filelist]

#     # Run tasks concurrently and gather results
#     results = await asyncio.gather(*tasks)

#     # Create a DataFrame from the results
#     dataset = pd.DataFrame(results, columns=columns.keys())

#     return dataset
