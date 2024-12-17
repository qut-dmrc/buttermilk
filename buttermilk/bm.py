# Configurations are stored in yaml files and managed by the Hydra library.
#
# Projects will  have a common config.yaml file that will be used to store configurations that
# are common to all the experiments in the project. Individual experiments will have their own
# config.yaml file that will be used to store configurations that are specific to that experiment.
# Authentication credentials are stored in secure cloud key/secret vaults on GCP, Azure, or AWS.
# The configuration files will be used to store the paths to the authentication credentials in
# the cloud vaults.

import datetime
import json
import logging
import sys
import os
from collections.abc import Sequence
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Optional,
    TypeVar,
)
from google import auth
import coloredlogs
import google.cloud.logging  # Don't conflict with standard logging
import humanfriendly
import pandas as pd
import pydantic
import shortuuid

from google.cloud import bigquery, storage
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from google.cloud import aiplatform
from omegaconf import DictConfig
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from rich import print as rprint

from buttermilk.exceptions import FatalError
from buttermilk.llms import LLMs
from buttermilk.utils.keys import SecretsManager

from ._core.config import CloudProviderCfg, RunCfg, Tracing
from ._core.flow import Flow
from ._core.log import logger
from ._core.types import SessionInfo
from .utils import save

CONFIG_CACHE_PATH = ".cache/buttermilk/models.json"
_MODELS_CFG_KEY = "models_secret"

# https://cloud.google.com/bigquery/pricing
GOOGLE_BQ_PRICE_PER_BYTE = 5 / 10e12  # $5 per tb.


_REGISTRY = {}
_CONFIG = "config"  # registry key for config object


class ConfigRegistry:
    _instance = None
    _config: DictConfig | None = None

    @classmethod
    def get_instance(cls):
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


T = TypeVar("T", bound="BM")


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
    # From https://py.iceberg.apache.org/reference/pyiceberg/utils/singleton/
    _instances: ClassVar[dict] = {}  # type: ignore

    def __new__(cls, *args, **kwargs):  # type: ignore
        key = cls.__name__
        if key not in _REGISTRY:
            _REGISTRY[key] = super().__new__(cls)
        return _REGISTRY[key]

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        """Prevent deep copy operations for singletons (code from IcebergRootModel)"""
        return self


class Project(BaseModel):
    name: str
    job: str
    connections: Sequence[str] = Field(default_factory=list)
    secret_provider: CloudProviderCfg = Field(default=None)
    save_dest: CloudProviderCfg = Field(default=None)
    logger: CloudProviderCfg = Field(default=None)
    pubsub: CloudProviderCfg = Field(default=None)
    clouds: list[CloudProviderCfg] = Field(default_factory=list)
    flows: list[Flow] = Field(default_factory=list)
    tracing: Tracing | None = Field(default_factory=Tracing)
    verbose: bool = True
    run: RunCfg

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = False
        populate_by_name = True
        exclude_none = True
        exclude_unset = True

    @model_validator(mode="after")
    def register(self) -> "Project":
        global _REGISTRY
        if not _REGISTRY.get(_CONFIG):
            _REGISTRY[_CONFIG] = self
        return self


class BM(Singleton, BaseModel):
    cfg: Optional[Project] = Field(default=None, validate_default=True)

    _run_metadata: SessionInfo = PrivateAttr(default_factory=SessionInfo)
    _gcp_project: str = PrivateAttr(default=None)
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @field_validator("cfg")
    def load_config(cls, v):
        global _REGISTRY
        if not v:
            try:
                return _REGISTRY[_CONFIG]
            except KeyError as e:
                raise FatalError(
                    "BM() called without config information before it was initialised.",
                ) from e
        return v

    def model_post_init(self, __context: Any) -> None:
        if not _REGISTRY.get("init"):
            for cloud in self.cfg.clouds:
                if cloud.type == "gcp":
                    # authenticate to GCP
                    os.environ['GOOGLE_CLOUD_PROJECT'] = os.environ.get('GOOGLE_CLOUD_PROJECT', cloud.project)
                    credentials, self._gcp_project = auth.default(quota_project_id=cloud.quota_project_id)
                    self._run_metadata.save_dir = f"gs://{cloud.bucket}/runs/{self._run_metadata.run_id}"
                    self.setup_logging(verbose=self.cfg.logger.verbose)
                    self.logger.info(f"Authenticated to gcloud using default credentials, project: {self._gcp_project}, save dir: {self._run_metadata.save_dir}") 
                if cloud.type == "vertex":
                    # initialize vertexai
                    aiplatform.init(
                        project=cloud.project,
                        location=cloud.region,
                        staging_bucket=cloud.bucket,
                    )
            _REGISTRY["init"] = True

            # Print config to console and save to default save dir
            rprint(self.cfg)
            rprint(self._run_metadata)

            try:
                save.save(
                    data=[self.cfg.model_dump(), self._run_metadata.model_dump()],
                    basename="config",
                    extension="json",
                    save_dir=self._run_metadata.save_dir,
                )

            except Exception as e:
                self.logger.error(f"Could not save config to default save dir: {e}")

            if self.cfg.tracing:
                from promptflow.tracing import start_trace
                start_trace(
                    resource_attributes={"run_id": self._run_metadata.run_id},
                    collection=self.cfg.name,
                    job=self.cfg.job,
                )
                
    def get_secret(
        self,
        secret_name: str = None,
        secret_class: str = None,
        cfg_key: str = None,
        version="latest",
    ):
        contents = self.secret_manager.get_secret(
            secret_name=secret_name,
            secret_class=secret_class,
            cfg_key=cfg_key,
            version=version,
        )

        return contents

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
        warnings.filterwarnings(
            action="ignore",
            module="msal",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            action="ignore",
            message="The `dict` method is deprecated",
            module="promptflow-tracing",
            category=DeprecationWarning,
        )

        console_format = "%(asctime)s %(hostname)s %(name)s %(filename).20s[%(lineno)4d] %(levelname)s %(message)s"
        if not verbose:
            coloredlogs.install(
                level="INFO",
                logger=logger,
                fmt=console_format,
                isatty=True,
                stream=sys.stdout,
            )
        else:
            coloredlogs.install(
                level="DEBUG",
                logger=logger,
                fmt=console_format,
                isatty=True,
                stream=sys.stdout,
            )

        # Labels for cloud logger
        resource = google.cloud.logging.Resource(
            type="generic_task",
            labels={
                "project_id": self.cfg.logger.project,
                "location": self.cfg.logger.location,
                "namespace": self.cfg.name,
                "job": self.cfg.job,
                "task_id": self._run_metadata.run_id,
            },
        )

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        _REGISTRY['gcslogging'] = google.cloud.logging.Client(project=self.cfg.logger.project)
        cloudHandler = CloudLoggingHandler(
            client=_REGISTRY['gcslogging'],
            resource=resource,
            name=self.cfg.name,
            labels=self._run_metadata.model_dump(),
        )
        cloudHandler.setLevel(
            logging.INFO,
        )  # Cloud logging never uses the DEBUG level, there's just too much data. Print debug to console only.
        logger.addHandler(cloudHandler)

        logger.info(
            dict(
                message=f"Logging setup for: {self._run_metadata.__str__}. Ready for data collection, saving log to Google Cloud Logs ({resource}). Default save directory for data in this run is: {self._run_metadata.save_dir}",
                **self._run_metadata.model_dump(),
            ),
        )

        try:
            from importlib.metadata import version

            logger.debug(f"Buttermilk version is: {version('buttermilk')}")
        except:
            pass

    @property
    def gcs(self) -> storage.Client:
        if _REGISTRY.get("gcs") is None:
            _REGISTRY["gcs"] = storage.Client(project=self._gcp_project)
        return _REGISTRY["gcs"]
    
    @property
    def secret_manager(self) -> SecretsManager:
        if _REGISTRY.get("secret_manager") is None:
            _REGISTRY["secret_manager"] = SecretsManager(**self.cfg.secret_provider.model_dump())
        return _REGISTRY["secret_manager"]

    def save(self, data, basename="", extension=".jsonl", **kwargs):
        """Failsafe save method."""
        result = save.save(data=data, save_dir=self._run_metadata.save_dir, basename=basename, extension=extension, **kwargs)
        logger.info(dict(message=f"Saved data to: {result}", uri=result, run_id=self._run_metadata.run_id))
        return result

    @property
    def llms(self) -> LLMs:
        if _REGISTRY.get("llms") is None:
            try:
                contents = Path(CONFIG_CACHE_PATH).read_text(encoding="utf-8")
                connections = json.loads(contents)
            except Exception:
                connections = self.get_secret(cfg_key=_MODELS_CFG_KEY)

            connections = {conn["name"]: conn for conn in connections}

            _REGISTRY["llms"] = LLMs(connections=connections)
        return _REGISTRY["llms"]

    @property
    def bq(self) -> bigquery.Client:
        if _REGISTRY.get("bq") is None:
            _REGISTRY["bq"] = bigquery.Client(project=self._gcp_project)
        return _REGISTRY["bq"]

    def run_query(
        self,
        sql,
        destination=None,
        overwrite=False,
        do_not_return_results=False,
        save_to_gcs=False,
        df=True,
    ) -> pd.DataFrame:
        t0 = datetime.datetime.now()

        job_config = {
            "use_legacy_sql": False,
        }

        # Cannot set write_disposition if saving to GCS
        if save_to_gcs:
            # Tell BigQuery to save the results to a specific GCS location
            gcs_results_uri = (
                f"{self._run_metadata.save_dir}/query_{shortuuid.uuid()}/*.json"
            )
            export_command = f"""   EXPORT DATA OPTIONS(
                        uri='{gcs_results_uri}',
                        format='JSON',
                        overwrite=false) AS """
            sql = export_command + sql
            self.logger.debug(f"Saving results to {gcs_results_uri}.")
        elif destination:
            self.logger.debug(f"Saving results to {destination}.")
            job_config["destination"] = destination
            job_config["allow_large_results"] = True

            if overwrite:
                job_config["write_disposition"] = "WRITE_TRUNCATE"
            else:
                job_config["write_disposition"] = "WRITE_APPEND"

        job_config = bigquery.QueryJobConfig(**job_config)
        job = self.bq.query(sql, job_config=job_config)

        bytes_billed = job.total_bytes_billed
        cache_hit = job.cache_hit

        if bytes_billed:
            approx_cost = bytes_billed * GOOGLE_BQ_PRICE_PER_BYTE
            bytes_billed = humanfriendly.format_size(bytes_billed)
            approx_cost = humanfriendly.format_number(approx_cost)
        else:
            approx_cost = "unknown"
        time_taken = datetime.datetime.now() - t0
        self.logger.info(
            f"Query stats: Ran in {time_taken} seconds, cache hit: {cache_hit}, billed {bytes_billed}, approx cost ${approx_cost}.",
        )

        if do_not_return_results:
            return True

        # job.result() blocks until the query has finished.
        result = job.result()
        if df:
            if result.total_rows > 0:
                results_df = result.to_dataframe()
                return results_df
            return pd.DataFrame()
        return result


if __name__ == "__main__":
    pass
