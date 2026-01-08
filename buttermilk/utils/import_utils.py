import importlib
from typing import Any


def load_class(class_path: str) -> Any:
    """Load a class from a dot-separated string path.

    Args:
        class_path: Path to class like 'module.submodule.ClassName'

    Returns:
        The class object

    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class not found in module
        ValueError: If class_path format is invalid
    """
    if "." not in class_path:
        raise ValueError(f"Invalid class path '{class_path}'. Must be 'module.ClassName'")

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
