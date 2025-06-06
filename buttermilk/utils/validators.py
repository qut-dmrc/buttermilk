# validators.py
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

import httpx
import pydantic
from bleach import clean
from cloudpathlib import CloudPath
from markdown_it import MarkdownIt
from omegaconf import DictConfig, ListConfig, OmegaConf

T = TypeVar("T")


def lowercase_validator(v: Any) -> str:
    if isinstance(v, str):
        return v.lower()
    return str(v).lower()


def uppercase_validator(v: Any) -> str:
    if isinstance(v, str):
        return v.upper()
    return str(v).upper()


def make_case_validator(style: Literal["upper", "lower", "sentence"] = "upper") -> Callable[[Any], str]:
    """Convert input to lowercase string if possible"""
    if style == "upper":
        return uppercase_validator
    if style == "lower":
        return lowercase_validator
    raise NotImplementedError
    return lowercase_validator


def make_list_validator() -> Callable[[Any], list]:
    """Convert single items to list if not already a list"""

    def validator(v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, ListConfig):
            # Convert Hydra's ListConfig to a regular list
            return list(v)
        return v if isinstance(v, list) else list(v)

    return validator


def convert_omegaconf_objects(v):
    """Recursively convert OmegaConf objects to standard Python types."""
    if isinstance(v, (DictConfig, ListConfig)):
        return OmegaConf.to_container(v, resolve=True)
    if isinstance(v, dict):
        return {k: convert_omegaconf_objects(v) for k, v in v.items()}
    if isinstance(v, list):
        return [convert_omegaconf_objects(item) for item in v]
    return v


def make_uri_validator() -> Callable[[Any], str]:
    """Convert input to string URI if possible"""

    def validator(path: Any) -> str:
        if isinstance(path, bytes):
            path = path.decode("utf-8")

        if isinstance(path, httpx.URL):
            return str(path)
        if isinstance(path, pydantic.AnyUrl):
            return str(path)
        if isinstance(path, CloudPath):
            return str(path.as_uri())
        if isinstance(path, Path):
            return str(path.as_posix())
        return path

    return validator


def make_path_validator() -> Callable[[Any], str]:
    """Convert CloudPath to string URI"""

    def validator(path: Any) -> str:
        if isinstance(path, CloudPath):
            return str(path.as_uri())
        return path

    return validator


def sanitize_html(value: str) -> str:
    """Sanitizes HTML input."""
    cleaned = clean(value, tags=[], attributes={}, strip=True)  # Allow no tags/attributes
    return cleaned


def sanitize_markdown(value: str) -> str:
    """Sanitizes Markdown, converting it to safe HTML."""
    md = MarkdownIt("commonmark", {"breaks": True, "html": True})
    html_output = md.render(value)
    return html_output
