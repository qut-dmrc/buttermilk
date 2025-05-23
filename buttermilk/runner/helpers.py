from collections.abc import Mapping, Sequence
from tempfile import NamedTemporaryFile

import cloudpathlib
import pandas as pd

from buttermilk import buttermilk as bm  # Global Buttermilk instance
from buttermilk._core.config import DataSourceConfig
from buttermilk._core.log import logger  # noqa
from buttermilk.utils.flows import col_mapping_hydra_to_local
from buttermilk.utils.utils import find_key_string_pairs, load_json_flexi


def load_data(data_cfg: DataSourceConfig) -> pd.DataFrame:
    from buttermilk import logger

    if data_cfg.type == "outputs":
        # add new data after the agent has run
        return pd.DataFrame()
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
        df = read_all_files(
            data_cfg.path,
            data_cfg.glob,
            columns=data_cfg.columns,
        )
    else:
        raise ValueError(f"Unknown data type: {data_cfg.type}")

    return df


def combine_datasets(
    existing_df: pd.DataFrame,
    datasources: list[DataSourceConfig] = [],
    results_df: pd.DataFrame = None,
) -> pd.DataFrame:
    if datasources:
        for src in datasources:
            if src.type == "outputs":
                new_df = results_df.copy()
            else:
                new_df = load_data(src)
            if not existing_df.empty:
                new_df = group_and_filter_jobs(
                    existing_dfs=existing_df,
                    data=new_df,
                    data_cfg=src,
                )
            if src.index:
                new_df.set_index(src.index)

            existing_df = new_df

    return existing_df


def load_bq(data_cfg: DataSourceConfig) -> pd.DataFrame:

    sql = f"SELECT * FROM `{data_cfg.path}`"
    sql += " ORDER BY RAND() "

    df = bm.run_query(sql)

    return df


def load_jobs(data_cfg: DataSourceConfig) -> pd.DataFrame:

    last_n_days = data_cfg.last_n_days

    sql = "SELECT "
    if data_cfg.columns:
        sql_cols = []
        for field, locator in data_cfg.columns.items():
            locator = locator.split(".", 1)
            if len(locator) > 1:
                sql_cols.append(
                    f'JSON_VALUE({locator[0]}, "$.{locator[1]}") as {field}',
                )
            else:
                sql_cols.append(f"{locator[0]} as {field}")

        sql += ", ".join(sql_cols)

    else:
        sql += " * "
    sql += f" FROM `{data_cfg.path}` jobs WHERE (JSON_VALUE(outputs, '$.error') IS NULL) AND "

    sql += f" TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL {last_n_days} DAY)) "
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
                multi_or = " OR ".join([f'{comparison_string} = "{term}"' for term in condition])
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

    df = bm.run_query(sql)

    return df


def group_and_filter_jobs(
    *,
    data: pd.DataFrame,
    data_cfg: DataSourceConfig,
    existing_dfs: pd.DataFrame | None = None,
    raise_on_error=True,
) -> pd.DataFrame:
    from buttermilk import logger

    # expand and rename columns if we need to
    pairs_to_expand = (
        list(find_key_string_pairs(data_cfg.join)) + list(find_key_string_pairs(data_cfg.group)) + list(find_key_string_pairs(data_cfg.columns))
    )

    for col_name, grp in pairs_to_expand:
        try:
            grp, col = grp.split(".", 1)

            # extract the contents of the nested source column that
            # will form the new index

            # First, check if the column is a JSON string, and interpret it.
            try:
                data[grp] = data[grp].apply(
                    lambda x: load_json_flexi(x) if pd.notna(x) else dict(),
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
    idx_cols = list(data_cfg.join.keys()) + list(data_cfg.group.keys()) + list(data_cfg.index)

    idx_cols = [c for c in idx_cols if c in data.columns]
    if idx_cols:
        data = data.set_index(idx_cols)  # , drop=False)

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
        data = data.sample(frac=1).groupby(level=idx_cols, as_index=True).agg(lambda x: x.tolist()[: data_cfg.max_records_per_group])

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
            data = data.reset_index(level=list(data_cfg.group.keys()), drop=False)

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


async def prepare_step_df(data_configs: Mapping[str, DataSourceConfig]) -> dict[str, pd.DataFrame]:
    # This works for small datasets that we can easily read and load.
    datasets = {}
    source_list = []

    # data_cfg is not ordered. Loop through and load the static data first.
    for key, src in data_configs.items():
        if src.type == "job":
            # end of list (dynamic data)
            source_list.append(key)
        else:
            # start of list
            source_list = [key] + source_list

    combined_df = pd.DataFrame()
    for key in source_list:
        src = data_configs[key]
        new_df = load_data(src)
        # Load and join prior job data
        new_df = group_and_filter_jobs(
            existing_dfs=combined_df,
            data=new_df,
            data_cfg=src,
        )

        if src.columns:
            new_df = new_df[src.columns.keys()]

        # shuffle
        new_df = new_df.sample(frac=1)

        # Also hold the combined data for next iteration
        combined_df = new_df.copy()

        datasets[key] = new_df
    return datasets


def read_all_files(uri, pattern, columns: dict[str, str]):
    from buttermilk import logger

    filelist = cloudpathlib.GSPath(uri).glob(pattern)
    # Read each file into a DataFrame and store in a list
    dataset = pd.DataFrame(columns=columns.keys())
    for file in filelist:
        logger.debug(f"Reading {file.name} from {file.parent}...")
        content = file.read_bytes().decode("utf-8")
        dataset.loc[len(dataset)] = (file.stem, content)
    return dataset


def parse_flow_vars(
    var_map: Mapping,
    *,
    flow_data: dict,
    additional_data: dict = {},
) -> dict:
    # Take an input map of variable names to a dot-separated JSON path.
    # Returns a dict of variables with their corresponding content, sourced from
    # datasets provided in additional_data or records in job.

    mapped_vars = {}

    # Add inputs from previous runs
    for key, value in additional_data.items():
        if key in flow_data:
            raise ValueError(f"Key {key} already exists in input dataset.")
        if isinstance(value, pd.DataFrame):
            flow_data[key] = value.to_dict(orient="records")
        else:
            flow_data[key] = value

    def resolve_var(*, match_key: str, data_dict: dict):
        """Find a key in dot notation from a hierarchical dict."""
        if not data_dict:
            return None

        # If the final data variable is a mapping, return the value at the key we
        # are looking for. But if the value is not found or the variable is not a
        # mapping, return None. Later, the mapping's value will be used as a
        # literal string.
        if isinstance(data_dict, Mapping) and match_key in data_dict:
            return data_dict[match_key]

        # If the current data var is a list, check each element
        if isinstance(data_dict, Sequence) and not isinstance(data_dict, str):
            return [resolve_var(match_key=match_key, data_dict=x) for x in data_dict]

        if "." in match_key:
            next_level, locator = match_key.split(".", maxsplit=1)
            return resolve_var(
                match_key=locator,
                data_dict=data_dict.get(next_level, {}),
            )
        return None

    def descend(map, path):
        if path is None:
            return None

        if isinstance(path, str):
            # We have reached the end of the tree, this last path is a plain string
            # Use this final leaf as the locator for the data to insert here
            value = resolve_var(match_key=path, data_dict=flow_data)
            # value = jq.all(path, all_data_sources)
            return value
        if isinstance(path, bool):
            return path
        if isinstance(path, Sequence):
            # The key here is made up of multiple records
            # Descend recurisvely and fill it out.
            value = []
            for x in path:
                sub_value = descend(map=map, path=x)
                if sub_value and isinstance(sub_value, Sequence) and not isinstance(sub_value, str):
                    value.extend(sub_value)
                elif sub_value:
                    value.append(sub_value)
                else:
                    # add the bare string
                    value.append(x)
            return value
        if isinstance(path, Mapping):
            # The data here is another layer of a key:value mapping
            # Descend recurisvely and fill it out.
            value = {k: descend(map=k, path=v) for k, v in path.items() if v}
            return value
        raise ValueError(f"Unknown type in map: {type(path)} @ {map}")

    if var_map:
        for var, locator in var_map.items():
            if isinstance(locator, str) and locator == "record":
                # Skip references to full records, they belong in placeholders later on.
                pass
            else:
                value = descend(var, locator)
                if value:
                    mapped_vars[var] = value
                elif locator:
                    # Instead, treat the value of the input mapping as a string and add in full
                    mapped_vars[var] = locator
                else:
                    # Don't add empty values to the input dict
                    pass
    return mapped_vars

    # # Handle direct record field reference
    # if job.record and (
    #     match_key in job.record.model_fields or match_key in job.record.model_extra
    # ):
    #     return getattr(job.record, match_key)

    # # handle entire dataset
    # if additional_data and match_key in additional_data:
    #     if isinstance(additional_data[match_key], pd.DataFrame):
    #         return additional_data[match_key].astype(str).to_dict(orient="records")

    #     found = additional_data[match_key]
    #     if isinstance(found, (DictConfig,ListConfig)):
    #         found = OmegaConf.to_object(found)
    #     return found
