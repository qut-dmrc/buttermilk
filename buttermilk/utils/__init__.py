from typing import Any

from .flows import col_mapping_hydra_to_local
from .utils import (
    download_limited,
    download_limited_async,
    extract_url,
    extract_url_regex,
    find_key_string_pairs,
    get_ip,
    get_pdf_text,
    make_serialisable,
    read_file,
    read_json,
    read_text,
    read_yaml,
    remove_punctuation,
    scrub_serializable,
)

__all__ = [
    "col_mapping_hydra_to_local",
    "construct_dict_from_schema",
    "download_limited",
    "download_limited_async",
    "extract_url",
    "extract_url_regex",
    "find_key_string_pairs",
    "get_ip",
    "get_pdf_text",
    "init_viz",
    "make_serialisable",
    "read_file",
    "read_json",
    "read_text",
    "read_yaml",
    "remove_punctuation",
    "scrub_serializable",
]


def __getattr__(name: str) -> Any:
    """Lazy import for heavy BigQuery utilities and optional viz dependencies."""
    if name == "construct_dict_from_schema":
        from .bq import construct_dict_from_schema

        return construct_dict_from_schema
    if name == "init_viz":
        from .viz import init_viz

        return init_viz
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
