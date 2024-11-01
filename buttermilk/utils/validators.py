# validators.py
from typing import Any, Callable, TypeVar
from cloudpathlib import CloudPath
from pydantic import BaseModel, field_validator

T = TypeVar("T")

def make_list_validator() -> Callable[[Any], list]:
    """Convert single items to list if not already a list"""
    def validator(v: Any) -> list:
        if isinstance(v, str):
            return [v]
        return v if isinstance(v, list) else [v]
    return validator

def make_path_validator() -> Callable[[Any], str]:
    """Convert CloudPath to string URI"""
    def validator(path: Any) -> str:
        if isinstance(path, CloudPath):
            return str(path.as_uri())
        return path
    return validator


