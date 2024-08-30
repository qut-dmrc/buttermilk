# Configurations are stored in yaml files and managed by the Hydra library.
#
# Projects will  have a common config.yaml file that will be used to store configurations that
# are common to all the experiments in the project. Individual experiments will have their own
# config.yaml file that will be used to store configurations that are specific to that experiment.
# Authentication credentials are stored in secure cloud key/secret vaults on GCP, Azure, or AWS.
# The configuration files will be used to store the paths to the authentication credentials in
# the cloud vaults.

import datetime
import itertools
import json
import logging
import os
import platform
import sys
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, Union

from google.cloud import bigquery, storage
import cloudpathlib
import coloredlogs
import fsspec
import google.cloud.logging  # Don't conflict with standard logging
import humanfriendly
import psutil
import pydantic
import requests
import shortuuid
import yaml
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from cloudpathlib import AnyPath, CloudPath
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, SCMode
from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr, field_validator,
                      model_validator, root_validator)
from promptflow.tracing import start_trace, trace
from  typing import  Self, MutableMapping
from .utils import save
import types
CONFIG_CACHE_PATH = ".cache/buttermilk/.models.json"

_LOGGER_NAME = "buttermilk"
_REGISTRY = {}

logger=getLogger(_LOGGER_NAME)

T = TypeVar("T", bound="BM")

def _convert_to_hashable_type(element: Any) -> Any:
    if isinstance(element, dict):
        return tuple((_convert_to_hashable_type(k), _convert_to_hashable_type(v)) for k, v in element.items())
    elif isinstance(element, list):
        return tuple(map(_convert_to_hashable_type, element))
    return element

class Singleton:
    ## From https://py.iceberg.apache.org/reference/pyiceberg/utils/singleton/
    _instances: ClassVar[Dict] = {}  # type: ignore

    def __new__(cls, *args, **kwargs):  # type: ignore
        key = cls.__qualname__
        if key not in _REGISTRY:
            _REGISTRY[key] = super().__new__(cls)
        return _REGISTRY[key]

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        """Prevent deep copy operations for singletons (code from IcebergRootModel)"""
        return self

class BM(Singleton, BaseModel):
    cfg: Optional[Any] = Field(default_factory=lambda: _REGISTRY.get('cfg'))
    _run_id: str = PrivateAttr(default_factory=lambda: BM.make_run_id())
    save_dir: Optional[str] = None
    _clients: dict[str, Any] = {}

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @field_validator('cfg', mode='before')
    def get_config(cls, v):
        if _REGISTRY.get('cfg'):
            if v:
                raise ValueError("Config passed in but we already have one loaded.")
            else:
                return _REGISTRY['cfg']
        elif v:
            _REGISTRY['cfg'] = v
            return v
        else:
            raise ValueError("No config passed in on first initialisation.")

    def model_post_init(self, __context: Any) -> None:
        if not _REGISTRY.get('init'):
            self.save_dir = self._get_save_dir(self.save_dir)
            self.setup_logging()
            start_trace(resource_attributes={"run_id": self._run_id}, collection=self.cfg['name'], job=self.cfg['job'])
            _REGISTRY['init'] = True


    @classmethod
    def make_run_id(cls) -> str:
        # Create a unique identifier for this run
        node_name = platform.uname().node
        username = psutil.Process().username()
        # get rid of windows domain if present
        username = str.split(username, "\\")[-1]

        # The ISO 8601 format has too many special characters for a filename, so we'll use a simpler format
        run_time = datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y%m%dT%H%MZ")

        run_id = f"{run_time}-{shortuuid.uuid()[:4]}-{node_name}-{username}"

        return run_id

    @cached_property
    def metadata(self):# -> dict[str, Any]:
        labels = {
            "function_name": self.cfg['name'],
            "job": self.cfg['job'],
            "logs": self._run_id,
            "user": psutil.Process().username(),
            "node": platform.uname().node
        }
        return labels


    @cached_property
    def _connections_azure(self) -> dict:
        # Model definitions are stored in Azure Secrets, and loaded here.

        try:
            contents = Path(CONFIG_CACHE_PATH).read_text()
        except Exception as e:
            pass
        auth = DefaultAzureCredential()
        vault_uri = self.cfg["project"]["azure"]["vault"]
        models_secret = self.cfg["project"]["models_secret"]
        secrets = SecretClient(vault_uri, credential=auth)
        contents = secrets.get_secret(models_secret).value
        if not contents:
            raise ValueError("Could not load secrets from Azure vault")
        try:
            Path(CONFIG_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
            Path(CONFIG_CACHE_PATH).write_text(contents)
        except:
            pass

        connections = json.loads(contents)
        connections = {conn["name"]: conn for conn in connections}

        return connections

    @property
    def logger(self) -> logging.Logger:
        return getLogger(_LOGGER_NAME)

    def setup_logging(
        self,
        verbose=False,
    ) -> None:

        # Quieten other loggers down a bit (particularly requests and google api client)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)

        for logger_str in list(logging.Logger.manager.loggerDict.keys()):
            try:
                logging.getLogger(logger_str).setLevel(logging.WARNING)
            except:
                pass


        logger = getLogger(_LOGGER_NAME)

        for handler in logger.handlers:
            logger.removeHandler(handler)

        console_format = "%(asctime)s %(hostname)s %(name)s %(filename).20s[%(lineno)4d] %(levelname)s %(message)s"
        if not verbose:
            coloredlogs.install(
                level="INFO", logger=root_logger, fmt=console_format, stream=sys.stdout
            )
        else:
            coloredlogs.install(
                level="DEBUG", logger=logger, fmt=console_format, stream=sys.stdout
            )

        # Labels for cloud logger
        resource = google.cloud.logging.Resource(
            type="generic_task",
            labels={
                "project_id": "dmrc-platforms",
                "location": "us-central1",
                "namespace": self.cfg['name'],
                "job": self.cfg['job'],
                "task_id": self._run_id,
            },
        )

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        client = google.cloud.logging.Client()
        cloudHandler = CloudLoggingHandler(
            client=client, resource=resource, name=self.cfg['name'], labels=self.metadata
        )
        cloudHandler.setLevel(
            logging.INFO
        )  # Cloud logging never uses the DEBUG level, there's just too much data. Print debug to console only.
        logger.addHandler(cloudHandler)

        logger.info(
            f"Logging setup for: {self.metadata}. Ready for data collection, saving log to Google Cloud Logs ({resource}). Default save directory for data in this run is: {self.save_dir}"
        )

        try:
            from importlib.metadata import version

            logger.debug(f"Buttermilk version is: {version('buttermilk')}")
        except:
            pass

    @property
    def gcs(self) -> storage.Client:
        if self._clients.get('gcs') is None:
            self._clients['gcs'] = storage.Client(project=self.cfg.project.gcp.project)
        return self._clients['gcs']

    def _get_save_dir(self, value=None) -> str:
        # Get the save directory from the configuration
        if value:
            if isinstance(value, str):
                save_dir = value
            elif isinstance(value, (Path)):
                save_dir = value.as_posix()
            elif isinstance(value, (CloudPath)):
                save_dir = value.as_uri()
            else:
                raise ValueError(
                    f"save_path must be a string, Path, or CloudPath, got {type(value)}"
                )
        else:
            save_dir = self.cfg['project'].get('save_dir')
            if not save_dir:
                save_dir = (
                    f"gs://{self.cfg['project']['gcp']['bucket']}/runs/{self.cfg['name']}/{self.cfg['job']}/{self._run_id}"
                )

        # # Make sure the save directory is a valid path
        try:
            _ = cloudpathlib.AnyPath(save_dir)
        except Exception as e:
            raise ValueError(f"Invalid cloud save directory: {save_dir}. Error: {e}")

        return save_dir

    def save(self, data, **kwargs):
        """ Failsafe save method."""
        result = save.save(data=data, save_dir=self.save_dir, **kwargs)
        logger.info(f"Saved data to: {result}")
        return result