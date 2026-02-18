# validators.py
from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import httpx
import pydantic
from bleach import clean
from markdown_it import MarkdownIt
from omegaconf import DictConfig, ListConfig, OmegaConf

# Lazy import cloudpathlib (pulls in google.cloud.storage)
if TYPE_CHECKING:
    pass

T = TypeVar("T")


def lowercase_validator(v: Any) -> str:
    if isinstance(v, str):
        return v.lower()
    return str(v).lower()


def uppercase_validator(v: Any) -> str:
    if isinstance(v, str):
        return v.upper()
    return str(v).upper()


def make_case_validator(
    style: Literal["upper", "lower", "sentence"] = "upper",
) -> Callable[[Any], str]:
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
        # Lazy import cloudpathlib (pulls in google.cloud.storage)
        from cloudpathlib import CloudPath

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
        # Lazy import cloudpathlib
        from cloudpathlib import CloudPath

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


def import_class_from_path(class_path: str, expected_base_class: type | None = None) -> type:
    """Import a class from a fully qualified module path.

    Args:
        class_path: Fully qualified path like "app.models.RightsExtraction"
        expected_base_class: Optional base class to verify the imported class inherits from

    Returns:
        The imported class

    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class cannot be found in module
        ValueError: If class_path format is invalid or class doesn't inherit from expected_base_class

    Examples:
        >>> cls = import_class_from_path("app.models.RightsExtraction")
        >>> from pydantic import BaseModel
        >>> cls = import_class_from_path("app.models.DBRDocument", BaseModel)
    """
    if not class_path or not isinstance(class_path, str):
        raise ValueError(f"class_path must be a non-empty string, got: {class_path}")

    # Parse the class path
    try:
        module_path, class_name = class_path.rsplit(".", 1)
    except ValueError as e:
        raise ValueError(f"Invalid class path format '{class_path}'. Expected 'module.path.ClassName'") from e

    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}' from class path '{class_path}'") from e

    # Get the class
    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"Module '{module_path}' has no attribute '{class_name}'") from e

    # Verify it's a class
    if not isinstance(cls, type):
        raise ValueError(f"'{class_path}' does not resolve to a class (got {type(cls).__name__})")

    # Verify inheritance if expected_base_class is provided
    if expected_base_class is not None:
        if not issubclass(cls, expected_base_class):
            raise ValueError(f"Class '{class_path}' is not a subclass of {expected_base_class.__name__}")

    return cls


def make_class_import_validator(
    expected_base_class: type | None = None,
) -> Callable[[Any], type]:
    """Create a validator that imports a class from a string path or passes through existing classes.

    This validator is designed for Pydantic fields that accept either:
    - A string class path like "app.models.RightsExtraction"
    - An already-imported class object

    Args:
        expected_base_class: Optional base class to verify the class inherits from

    Returns:
        A validator function suitable for use with Pydantic's field_validator

    Examples:
        >>> from pydantic import BaseModel, Field, field_validator
        >>> class AgentConfig(BaseModel):
        ...     output_model: type[BaseModel] | None = Field(default=None)
        ...     _validate_output_model = field_validator("output_model", mode="before")(
        ...         make_class_import_validator(BaseModel)
        ...     )
    """

    def validator(v: Any) -> type | None:
        if v is None:
            return None

        # If already a class, verify and return
        if isinstance(v, type):
            if expected_base_class is not None and not issubclass(v, expected_base_class):
                raise ValueError(f"Class {v.__name__} is not a subclass of {expected_base_class.__name__}")
            return v

        # If string, import the class
        if isinstance(v, str):
            return import_class_from_path(v, expected_base_class)

        raise ValueError(f"output_model must be a class or string class path, got {type(v).__name__}")

    return validator
