"""Unified storage framework for Buttermilk.

This module provides a unified approach to data storage operations,
supporting both reading and writing with the same configuration and classes.
"""

from .._core.storage_config import StorageConfig
from .base import Storage
from .bigquery import BigQueryStorage
from .file import FileStorage

__all__ = [
    "StorageConfig",
    "Storage", 
    "BigQueryStorage",
    "FileStorage",
]