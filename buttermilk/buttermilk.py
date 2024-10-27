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
import types
from dataclasses import dataclass
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    MutableMapping,
    Optional,
    Self,
    Type,
    TypeVar,
    Union,
)

import cloudpathlib
import coloredlogs
import fsspec
import google.cloud.logging  # Don't conflict with standard logging
import humanfriendly
import pandas as pd
import psutil
import pydantic
import requests
import shortuuid
import yaml
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from cloudpathlib import AnyPath, CloudPath
from dotenv import load_dotenv
from google.cloud import bigquery, storage
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, SCMode
from promptflow.tracing import start_trace, trace
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
    root_validator,
)

from .utils import save, get_ip

CONFIG_CACHE_PATH = ".cache/buttermilk/models.json"

# https://cloud.google.com/bigquery/pricing
GOOGLE_BQ_PRICE_PER_BYTE = 5 / 10e12  # $5 per tb.

from buttermilk.utils.log import logger

_REGISTRY = {}

_ = load_dotenv()

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
        key = cls.__name__
        if key not in _REGISTRY:
            _REGISTRY[key] = super().__new__(cls)
        return _REGISTRY[key]

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        """Prevent deep copy operations for singletons (code from IcebergRootModel)"""
        return self

class SessionInfo(BaseModel):
    project: str
    run_id: str = Field(default_factory=lambda: SessionInfo.make_run_id)

    ip: str = Field(default_factory=get_ip)
    node_name: str = Field(default_factory=lambda: platform.uname().node)
    username: str = Field(default_factory=lambda: psutil.Process().username().split("\\")[-1])

    save_dir: Optional[str] = ''

    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, populate_by_name=True
    )

    @pydantic.model_validator(mode="after")
    def get_save_dir(self) -> Self:
        # Get the save directory from the configuration
        if self.save_dir:
            if isinstance(self.save_dir, str):
                save_dir = self.save_dir
            elif isinstance(self.save_dir, (Path)):
                save_dir = self.save_dir.as_posix()
            elif isinstance(self.save_dir, (CloudPath)):
                save_dir = self.save_dir.as_uri()
            else:
                raise ValueError(
                    f"save_path must be a string, Path, or CloudPath, got {type(self.save_dir)}"
                )
        else:
            if save_dir := self.cfg.get('save_dir'):
                save_dir = (
                    f"gs://{self.cfg.gcp.bucket}/runs/{self.project}/{self.run_id}"
                )

        return self

        # # Make sure the save directory is a valid path
        try:
            _ = cloudpathlib.AnyPath(save_dir)
        except Exception as e:
            raise ValueError(f"Invalid cloud save directory: {save_dir}. Error: {e}")

        return save_dir

    @classmethod
    def make_run_id(cls) -> str:
        global _REGISTRY
        if run_id := _REGISTRY.get("run_id"):
            return run_id

        # Create a unique identifier for this run
        node_name = platform.uname().node
        username = psutil.Process().username()
        # get rid of windows domain if present
        username = str.split(username, "\\")[-1]

        # The ISO 8601 format has too many special characters for a filename, so we'll use a simpler format
        run_time = datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y%m%dT%H%MZ")

        run_id = f"{run_time}-{shortuuid.uuid()[:4]}-{node_name}-{username}"
        _REGISTRY["run_id"] = run_id
        return run_id
    
class BM(Singleton, BaseModel):

    cfg: Any = Field(default_factory=dict, validate_default=True)
    _clients: dict[str, Any] = {}
    _run_info: SessionInfo = PrivateAttr()

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @field_validator('cfg', mode='before')
    def get_config(cls, v):
        if 'project' in v.keys():
            v = v.project
        if _REGISTRY.get('cfg'):
            if v:
                logger.debug("Config passed in but we already have one loaded. Overwriting.")
                _REGISTRY['cfg'] = v

            return _REGISTRY['cfg']
        elif v:
            _REGISTRY['cfg'] = v
            return v
        else:
            with initialize(version_base=None, config_path="conf"):
                v = compose(config_name="config").project
            _REGISTRY['cfg'] = v
            return v


    def model_post_init(self, __context: Any) -> None:
        from buttermilk.runner import RunInfo

        save_dir = self._get_save_dir()
        self._run_info = SessionInfo( project=self.cfg.name, job=self.cfg.job, save_dir=save_dir)

        if not _REGISTRY.get('init'):
            self.setup_logging(verbose=self.cfg.verbose)
            # start_trace(resource_attributes={"run_id": self._run_info.run_id}, collection=self.cfg.name, job=self.cfg.job)
            _REGISTRY['init'] = True
        


    @cached_property
    def metadata(self):# -> dict[str, Any]:
        labels = {
            "function_name": self.cfg.name,
            "job": self.cfg.job,
            "logs": self._run_info.run_id,
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
        vault_uri = self.cfg["azure"]["vault"]
        models_secret = self.cfg["models_secret"]
        secrets = SecretClient(vault_uri, credential=auth)
        contents = secrets.get_secret(models_secret).value
        if not contents:
            raise ValueError("Could not load secrets from Azure vault")
        try:
            Path(CONFIG_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
            Path(CONFIG_CACHE_PATH).write_text(contents)
        except Exception as e:
            pass

        connections = json.loads(contents)
        connections = {conn["name"]: conn for conn in connections}

        return connections

    @property
    def logger(self) -> logging.Logger:
        global logger
        return logger

    def setup_logging(
        self,
        verbose=False,
    ) -> None:
        global logger

        # Quieten other loggers down a bit (particularly requests and google api client)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)

        for logger_str in list(logging.Logger.manager.loggerDict.keys()):
            try:
                logging.getLogger(logger_str).setLevel(logging.WARNING)
            except:
                pass

        # Turn off some particularly annoying (and unresolvable) warnings generated by upstream libraries
        import warnings

        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
        warnings.filterwarnings(action="ignore",module="msal", category=DeprecationWarning)
        warnings.filterwarnings(action="ignore",message="The `dict` method is deprecated",module="promptflow-tracing", category=DeprecationWarning)

        console_format = "%(asctime)s %(hostname)s %(name)s %(filename).20s[%(lineno)4d] %(levelname)s %(message)s"
        if not verbose:
            coloredlogs.install(
                level="INFO", logger=logger, fmt=console_format, isatty=True, stream=sys.stdout
            )
        else:
            coloredlogs.install(
                level="DEBUG", logger=logger, fmt=console_format, isatty=True, stream=sys.stdout
            )

        # Labels for cloud logger
        resource = google.cloud.logging.Resource(
            type="generic_task",
            labels={
                "project_id": "dmrc-platforms",
                "location": "us-central1",
                "namespace": self.cfg.name,
                "job": self.cfg.job,
                "task_id": self._run_info.run_id,
            },
        )

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        client = google.cloud.logging.Client()
        cloudHandler = CloudLoggingHandler(
            client=client, resource=resource, name=self.cfg.name, labels=self.metadata
        )
        cloudHandler.setLevel(
            logging.INFO
        )  # Cloud logging never uses the DEBUG level, there's just too much data. Print debug to console only.
        logger.addHandler(cloudHandler)

        logger.info(
            dict(message=f"Logging setup for: {self.metadata}. Ready for data collection, saving log to Google Cloud Logs ({resource}). Default save directory for data in this run is: {self._run_info.save_dir}",
                save_dir=self._run_info.save_dir, **self.metadata)
        )

        try:
            from importlib.metadata import version

            logger.debug(f"Buttermilk version is: {version('buttermilk')}")
        except:
            pass

    @property
    def gcs(self) -> storage.Client:
        if self._clients.get('gcs') is None:
            self._clients['gcs'] = storage.Client(project=self.cfg.gcp.project)
        return self._clients['gcs']

    def save(self, data, basename='', extension='.jsonl', **kwargs):
        """ Failsafe save method."""
        result = save.save(data=data, save_dir=self._run_info.save_dir, basename=basename, extension=extension, **kwargs)
        logger.info(dict(message=f"Saved data to: {result}", uri=result, run_id=self._run_info.run_id))
        return result

    @property
    def bq(self) -> bigquery.Client:
        if self._clients.get('bq') is None:
            self._clients['bq'] = bigquery.Client(project=self.cfg.gcp.project)
        return self._clients['bq']

    def run_query(
        self,
        sql,
        destination=None,
        overwrite=False,
        do_not_return_results=False,
        save_to_gcs=False,
    ) -> pd.DataFrame:
        
        t0 = datetime.datetime.now()

        job_config = {
            "use_legacy_sql": False,
        }

        # Cannot set write_disposition if saving to GCS
        if save_to_gcs:
            # Tell BigQuery to save the results to a specific GCS location
            gcs_results_uri = f"{self._run_info.save_dir}/query_{shortuuid.uuid()}/*.json"
            export_command = f"""   EXPORT DATA OPTIONS(
                        uri='{gcs_results_uri}',
                        format='JSON',
                        overwrite=false) AS """
            sql = export_command + sql
            self.logger.debug(f"Saving results to {gcs_results_uri}.")
        else:
            if destination:
                self.logger.debug(f"Saving results to {destination}.")
                job_config["destination"] = destination
                job_config["allow_large_results"] = True

                if overwrite:
                    job_config["write_disposition"] = "WRITE_TRUNCATE"
                else:
                    job_config["write_disposition"] = "WRITE_APPEND"

        job_config = bigquery.QueryJobConfig(**job_config)
        job = self.bq.query(sql, job_config=job_config)

        result = job.result()  # blocks until job is finished

        bytes_billed = job.total_bytes_billed
        cache_hit = job.cache_hit

        approx_cost = bytes_billed * GOOGLE_BQ_PRICE_PER_BYTE
        bytes_billed = humanfriendly.format_size(bytes_billed)

        time_taken = datetime.datetime.now() - t0
        self.logger.info(
            f"Query stats: Ran in {time_taken} seconds, cache hit: {cache_hit}, billed {bytes_billed}, approx cost ${approx_cost:0.2}."
        )

        if do_not_return_results:
            return True
        else:
            # job.result() blocks until the query has finished.
            result = job.result()
            results_df = result.to_dataframe()
            return results_df
