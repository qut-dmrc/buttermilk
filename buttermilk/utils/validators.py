# validators.py
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import httpx
from omegaconf import DictConfig, ListConfig, OmegaConf
import pydantic
from cloudpathlib import CloudPath

T = TypeVar("T")


def make_list_validator() -> Callable[[Any], list]:
    """Convert single items to list if not already a list"""

    def validator(v: Any) -> list:
        if isinstance(v, str):
            return [v]
        return v if isinstance(v, list) else [v]

    return validator

def convert_omegaconf_objects() -> Callable[[Any], dict|list]:
    """Convert OmegaConf items to python objects"""

    def validator(v: Any) -> list|dict:
        if isinstance(v, (DictConfig,ListConfig)):
            return OmegaConf.to_container(v, resolve=True)
        return v

    return validator

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
