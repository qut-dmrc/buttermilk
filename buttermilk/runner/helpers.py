from collections.abc import Mapping, Sequence
from tempfile import NamedTemporaryFile

import cloudpathlib
import pandas as pd


def cache_data(uri: str) -> str:
    with NamedTemporaryFile(delete=False, suffix=".jsonl", mode="wb") as f:
        dataset = f.name
        data = cloudpathlib.CloudPath(uri).read_bytes()
        f.write(data)
    return dataset


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
