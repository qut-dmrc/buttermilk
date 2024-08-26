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
from omegaconf import DictConfig, OmegaConf
from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr,
                      model_validator, root_validator)

CONFIG_CACHE_PATH = ".cache/buttermilk/.models.json"

_LOGGER_NAME = "buttermilk"

T = TypeVar("T", bound="BM")
_REGISTRY = {}

class BM(BaseModel):

    _run_id: str = PrivateAttr(default_factory=lambda: BM.make_run_id())
    _instance: ClassVar[Dict[str, "BM"]] = {}

    _cfg: Any = PrivateAttr(default_factory=lambda: BM.get_config())

    save_dir: Optional[str] = None

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Create the singleton instance."""
        if cls.__name__ not in _REGISTRY:
            super().__init_subclass__(**kwargs)
            _REGISTRY[cls.__name__] = cls

    @model_validator(mode="before")
    @classmethod
    def instance(cls: Type[T], data: Any) -> T:
        """Get the singleton instance."""
        if cls.__name__ in _REGISTRY:
            return _REGISTRY[cls.__name__]
        else:
            return data

    def __repr__(self):
        return f"Singleton(name={self._cfg['name']}, job={self._cfg['job']}, run_id={self._run_id})"

    def model_post_init(self, __context: Any) -> None:
        #_REGISTRY[self.__name__] = self
        self.save_dir = self._get_save_dir(self.save_dir)
        self.setup_logging()

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

    @classmethod
    def get_config(
        cls, config_dir: str = "conf", config_name: str = "config"
    ) -> DictConfig:
        """Load the configuration from the given directory and file.

        Args:
            config_dir: The directory containing the configuration file.
            config_name: The name of the configuration file.

        Returns:
            The configuration dictionary.
        """
        with initialize(config_path=config_dir, version_base="1.3"):
            cfg = compose(config_name=config_name)
        return cfg

    @cached_property
    def metadata(self):# -> dict[str, Any]:
        labels = {
            "function_name": self._cfg['name'],
            "job": self._cfg['job'],
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
        vault_uri = self._cfg["project"]["azure"]["vault"]
        models_secret = self._cfg["project"]["models_secret"]
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
        logger = getLogger(_LOGGER_NAME)

        console_format = "%(asctime)s %(hostname)s %(name)s %(filename).20s[%(lineno)4d] %(levelname)s %(message)s"
        if not verbose:
            coloredlogs.install(
                level="INFO", logger=logger, fmt=console_format, stream=sys.stdout
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
                "namespace": self._cfg['name'],
                "job": self._cfg['job'],
                "task_id": self._run_id,
            },
        )

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        client = google.cloud.logging.Client()
        cloudHandler = CloudLoggingHandler(
            client=client, resource=resource, name=self._cfg['name'], labels=self.metadata
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
            save_dir = self._cfg["project"]["save_dir"]
            if not save_dir:
                save_dir = (
                    f"gs://{self._cfg['project']['gcp']['bucket']}/runs/{self._cfg['name']}/{self._cfg['job']}/{self._run_id}"
                )

        # # Make sure the save directory is a valid path
        try:
            _ = cloudpathlib.AnyPath(save_dir)
        except Exception as e:
            raise ValueError(f"Invalid cloud save directory: {save_dir}. Error: {e}")

        return save_dir