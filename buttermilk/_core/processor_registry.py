"""Processor registry for dynamic processor instantiation.

This module provides a registry pattern for mapping processor config types to their
corresponding processor classes. This enables dynamic loading of processors without
hardcoded if/elif chains.

The registry is populated at import time when processor modules are loaded, allowing
processors to self-register using the register_processor function.
"""

from typing import Type

from buttermilk._core.processor_config import ProcessorConfig
from buttermilk._core.protocols import BatchProcessor, Processor

# Registry mapping config type -> processor class
_PROCESSOR_REGISTRY: dict[str, Type[Processor | BatchProcessor]] = {}


def register_processor(
    config_type: str, processor_class: Type[Processor | BatchProcessor]
) -> None:
    """Register a processor class for a config type.

    Args:
        config_type: The processor type string (e.g., 'expander', 'groupchat')
        processor_class: The processor class to instantiate for this type

    Raises:
        ValueError: If config_type is already registered (fail-fast on conflicts)
    """
    if config_type in _PROCESSOR_REGISTRY:
        raise ValueError(
            f"Processor type '{config_type}' is already registered. "
            f"Existing: {_PROCESSOR_REGISTRY[config_type]}, "
            f"New: {processor_class}"
        )
    _PROCESSOR_REGISTRY[config_type] = processor_class


def create_processor(config: ProcessorConfig) -> Processor | BatchProcessor:
    """Create a processor instance from config.

    Args:
        config: The processor configuration

    Returns:
        Instantiated processor of the appropriate type

    Raises:
        KeyError: If processor type is not registered (fail-fast on unknown types)
    """
    processor_class = _PROCESSOR_REGISTRY[config.type]  # Fail-fast: KeyError if not found
    return processor_class(config)  # type: ignore


def get_registered_types() -> list[str]:
    """Return all registered processor types.

    Returns:
        List of registered processor type strings
    """
    return list(_PROCESSOR_REGISTRY.keys())
