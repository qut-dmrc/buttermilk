# Configurations are stored in yaml files and managed by the Hydra library.
#
# Projects will  have a common config.yaml file that will be used to store configurations that
# are common to all the experiments in the project. Individual experiments will have their own
# config.yaml file that will be used to store configurations that are specific to that experiment.
# Authentication credentials are stored in secure cloud key/secret vaults on GCP, Azure, or AWS.
# The configuration files will be used to store the paths to the authentication credentials in
# the cloud vaults.

from __future__ import annotations  # Enable postponed annotations

import threading
from typing import (
    Any,
    ClassVar,
    TypeVar,
)

from omegaconf import DictConfig

# Add a lock for thread safety
_singleton_lock = threading.Lock()


class ConfigRegistry:
    _instance = None
    _config: DictConfig | None = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # Use the lock to ensure thread-safe creation
            with _singleton_lock:
                # Check again inside the lock in case another thread created it
                # while waiting for the lock
                if cls._instance is None:
                    cls._instance = ConfigRegistry()
        return cls._instance

    @classmethod
    def set_config(cls, cfg: DictConfig):
        instance = cls.get_instance()
        instance._config = cfg

    @classmethod
    def get_config(cls) -> DictConfig | None:
        return cls.get_instance()._config


T = TypeVar("T")


def _convert_to_hashable_type(element: Any) -> Any:
    if isinstance(element, dict):
        return tuple(
            (_convert_to_hashable_type(k), _convert_to_hashable_type(v))
            for k, v in element.items()
        )
    if isinstance(element, list):
        return tuple(map(_convert_to_hashable_type, element))
    return element


class Singleton:
    # Use a class variable for instances, as suggested by the comment
    _instances: ClassVar[dict] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()  # Use a lock per Singleton class

    def __new__(cls, *args, **kwargs):
        # Use the lock to ensure thread-safe creation
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[cls] = instance
                # Call __init__ manually only on the first creation
                # This prevents __init__ from being called on subsequent calls
                instance.__init__(*args, **kwargs)
            return cls._instances[cls]

    def __deepcopy__(self, memo: dict[int, Any] | None = None):
        """Prevent deep copy operations for singletons"""
        return self
