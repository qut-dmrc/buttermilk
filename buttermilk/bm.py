# Configurations are stored in yaml files and managed by the Hydra library.
#
# Projects will  have a common config.yaml file that will be used to store configurations that
# are common to all the experiments in the project. Individual experiments will have their own
# config.yaml file that will be used to store configurations that are specific to that experiment.
# Authentication credentials are stored in secure cloud key/secret vaults on GCP, Azure, or AWS.
# The configuration files will be used to store the paths to the authentication credentials in
# the cloud vaults.

from __future__ import annotations  # Enable postponed annotations

import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Self,
    TypeVar,
)
import weave
from weave.trace.weave_client import WeaveClient
import coloredlogs
import google.cloud.logging  # Don't conflict with standard logging
import humanfriendly
import pandas as pd
import pydantic
import shortuuid
from google.auth.credentials import Credentials as GoogleCredentials
from google.cloud import aiplatform, bigquery, storage
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from omegaconf import DictConfig
from pydantic import (
    PrivateAttr,
    model_validator,
)

try:
    pass
except:
    pass

from ._core.config import Project
from ._core.llms import LLMs
from .utils.keys import SecretsManager
from .utils.utils import load_json_flexi

from ._core.log import logger
from .utils import save

CONFIG_CACHE_PATH = ".cache/buttermilk/models.json"
_MODELS_CFG_KEY = "models_secret"
_SHARED_CREDENTIALS_KEY = "credentials_secret"

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
    # From https://py.iceberg.apache.org/reference/pyiceberg/utils/singleton/
    _instances: ClassVar[dict] = {}  # type: ignore

    def __new__(cls, *args, **kwargs):  # type: ignore
        key = cls.__name__
        if key not in _REGISTRY:
            _REGISTRY[key] = super().__new__(cls)
        return _REGISTRY[key]

    def __deepcopy__(self, memo: dict[int, Any] | None = None):
        """Prevent deep copy operations for singletons"""
        return self


class BM(Singleton, Project):
    _gcp_project: str = PrivateAttr()
    _gcp_credentials_cached: GoogleCredentials = PrivateAttr(default=None)
    _weave: WeaveClient = PrivateAttr()

    @property
    def weave(self) -> WeaveClient:
        if not hasattr(self, "_weave"):
            self._weave = weave.init(f"{self.run_info.name}-{self.run_info.job}")
        return self._weave

    @property
    def _gcp_credentials(self) -> GoogleCredentials:
        from google.auth import default, transport
        billing_project = os.environ.get("google_billing_project", os.environ["GOOGLE_CLOUD_PROJECT"])

        if not self._gcp_credentials_cached:
            self._gcp_credentials_cached, self._gcp_project = default(
                quota_project_id=billing_project,
            )

        # GCP tokens last 60 minutes and need to be refreshed after that
        auth_request = transport.requests.Request()
        self._gcp_credentials_cached.refresh(auth_request)
        # self._gcp_token = credentials.refresh(google.auth.transport.requests.Request())

        return self._gcp_credentials_cached

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",  # Allow extra fields
    )

    @property
    def logger(self) -> logging.Logger:
        return logger

    @model_validator(mode="before")
    @classmethod
    def get_vars(cls, vars) -> dict:
        # Not sure why this does nothing... maybe debugging?
        return vars

    @model_validator(mode="after")
    def ensure_config(self) -> Self:
        for cloud in self.clouds:
            if cloud.type == "gcp":
                # store authentication info
                os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get(
                    "GOOGLE_CLOUD_PROJECT",
                    cloud.project,
                )
                if "quota_project_id" in cloud.model_fields_set:
                    os.environ["google_billing_project"] = cloud.quota_project_id

                # authenticate here
                _ = self._gcp_credentials

            if cloud.type == "vertex":
                # initialize vertexai
                aiplatform.init(
                    project=cloud.project,
                    location=cloud.location,
                    staging_bucket=cloud.bucket,
                )
                # list available models
                models = aiplatform.Model.list()

        if self.logger_cfg:
            self.setup_logging(verbose=self.logger_cfg.verbose)

            # Print config to console and save to default save dir
            print(self.model_dump())
            try:
                self.save(
                    data=[self.model_dump(), self.run_info.model_dump()],
                    basename="config",
                    extension=".json",
                    save_dir=self.save_dir,
                )

            except Exception as e:
                logger.error(f"Could not save config to default save dir: {e}")

        if self.tracing and self.run_info:
            collection = f"{self.run_info.name}-{self.run_info.job}"
            if self.tracing.provider == "weave":
                self._weave = weave.init(collection)
            elif self.tracing.provider == "traceloop":
                from traceloop.sdk import Traceloop

                Traceloop.init(
                    disable_batch=True,
                    api_key=self.tracing.api_key,
                )
            elif self.tracing.provider == "promptflow":
                from promptflow.tracing import start_trace

                start_trace(
                    resource_attributes={"run_id": self.run_info.run_id},
                    collection=collection,
                )

        return self

    @property
    def save_dir(self) -> str:
        return self.run_info.save_dir

    def save(self, data, save_dir=None, extension=None, **kwargs):
        """Failsafe save method."""
        save_dir = save_dir or self.save_dir
        result = save.save(data=data, save_dir=save_dir, extension=extension, **kwargs)
        logger.info(
            dict(
                message=f"Saved data to: {result}",
                uri=result,
                run_id=self.run_info.run_id,
            ),
        )
        return result

    def setup_logging(
        self,
        verbose=False,
    ) -> None:

        root_logger = logging.getLogger()

        # Remove existing handlers from the root logger to avoid duplicates
        # or conflicts if setup_logging is called multiple times or by other libraries.
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # for clarity
        from ._core.log import logger as bm_logger

        console_format = "%(asctime)s %(hostname)s %(name)s %(filename).20s[%(lineno)4d] %(levelname)s %(message)s"
        coloredlogs.install(
            logger=bm_logger,
            fmt=console_format,
            isatty=True,
            stream=sys.stderr,
        )

        resource = None
        if self.logger_cfg.type == "gcp":
            # Labels for cloud logger
            resource = google.cloud.logging.Resource(
                type="generic_task",
                labels={
                    "project_id": self.logger_cfg.project,
                    "location": self.logger_cfg.location,
                    "namespace": self.run_info.name,
                    "job": self.run_info.job,
                    "task_id": self.run_info.run_id,
                },
            )

            cloudHandler = CloudLoggingHandler(
                client=self.gcs_log_client,
                resource=resource,
                name=self.run_info.name,
                labels=self.run_info.model_dump(
                    include={"run_id", "name", "job", "platform", "username"},
                ),
            )
            cloudHandler.setLevel(
                logging.INFO,
            )  # Cloud logging never uses the DEBUG level, there's just too much data. Print debug to console only.
            bm_logger.addHandler(cloudHandler)

        # --- Set logging level: warning for others, either debug or info for us ---
        root_logger.setLevel(logging.WARNING)
        for logger_str in list(logging.Logger.manager.loggerDict.keys()):
            try:
                logging.getLogger(logger_str).setLevel(logging.WARNING)
            except:
                pass

        if verbose:
            bm_logger.setLevel(logging.DEBUG)
        else:
            bm_logger.setLevel(logging.INFO)

        # --- Quieten overly verbose libraries directly ---

        logging.getLogger("googleapiclient").setLevel(logging.WARNING)
        logging.getLogger("google.auth").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        # Add any other libraries that are too noisy

        # Turn off some particularly annoying (and unresolvable) warnings generated by upstream libraries
        import warnings

        warnings.filterwarnings(
            action="ignore",
            message="unclosed",
            category=ResourceWarning,
        )
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
        warnings.filterwarnings(
            action="ignore",
            module="traceloop",
            category=DeprecationWarning,
        )
        # warnings.filterwarnings(
        #     action="ignore",
        #     message="Passing field metadata as keyword arguments is deprecated",
        # )
        warnings.filterwarnings(
            action="ignore",
            message="`sentry_sdk.Hub` is deprecated",
            category=DeprecationWarning,
        )
        # warnings.filterwarnings(
        #     action="ignore",
        #     message="Support for class-based `config` is deprecated",
        #     category=DeprecationWarning,
        # )
        warnings.filterwarnings(
            action="ignore",
            module="marshmallow",
            category=Warning,
        )
        warnings.filterwarnings(
            action="ignore",
            message="jsonschema.RefResolver is deprecated",
            # category=DeprecationWarning,
            module="flask_restx",
        )
        warnings.filterwarnings(
            action="ignore",
            message="CropBox missing from",
        )

        message = (
            f"Logging set up for: {self.run_info.__str__()}. Ready for data collection. Default save directory for data in this run is: {self.save_dir}",
        )

        if resource:
            message = f"{message} {resource}"

        bm_logger.info(message, extra=dict(run=self.run_info.model_dump()))

        try:
            from importlib.metadata import version

            bm_logger.debug(f"Buttermilk version is: {version('buttermilk')}")
        except:
            pass

    @property
    def gcs_log_client(self) -> google.cloud.logging.Client:
        if _REGISTRY.get("gcslogging") is None:
            _REGISTRY["gcslogging"] = google.cloud.logging.Client(
                project=self.logger_cfg.project,
            )
        return _REGISTRY["gcslogging"]

    @property
    def gcs(self) -> storage.Client:
        if _REGISTRY.get("gcs") is None:
            _REGISTRY["gcs"] = storage.Client(project=self._gcp_project)
        return _REGISTRY["gcs"]

    @property
    def secret_manager(self) -> SecretsManager:
        if _REGISTRY.get("secret_manager") is None:
            _REGISTRY["secret_manager"] = SecretsManager(
                **self.secret_provider.model_dump(),
            )
        return _REGISTRY["secret_manager"]

    @property
    def credentials(self) -> dict[str, str]:
        if _REGISTRY.get("credentials") is None:
            _REGISTRY["credentials"] = self.secret_manager.get_secret(
                cfg_key=_SHARED_CREDENTIALS_KEY,
            )
        return _REGISTRY["credentials"]

    @property
    def llms(self) -> LLMs:
        if _REGISTRY.get("llms") is None:
            try:
                contents = Path(CONFIG_CACHE_PATH).read_text(encoding="utf-8")
                connections = load_json_flexi(contents)
            except Exception:
                connections = self.secret_manager.get_secret(
                    cfg_key=_MODELS_CFG_KEY,
                )
                try:
                    Path(CONFIG_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
                    Path(CONFIG_CACHE_PATH).write_text(
                        json.dumps(connections),
                        encoding="utf-8",
                    )
                except Exception as e:
                    logger.error(f"Unable to cache connections: {e}, {e.args}")

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
        return_df=True,
    ) -> pd.DataFrame:
        t0 = datetime.datetime.now()

        job_config = {
            "use_legacy_sql": False,
        }

        # Cannot set write_disposition if saving to GCS
        if save_to_gcs:
            # Tell BigQuery to save the results to a specific GCS location
            gcs_results_uri = f"{self.save_dir}/query_{shortuuid.uuid()}/*.json"
            export_command = f"""   EXPORT DATA OPTIONS(
                        uri='{gcs_results_uri}',
                        format='JSON',
                        overwrite=false) AS """
            sql = export_command + sql
            logger.debug(f"Saving results to {gcs_results_uri}.")
        elif destination:
            logger.debug(f"Saving results to {destination}.")
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
        logger.info(
            f"Query stats: Ran in {time_taken} seconds, cache hit: {cache_hit}, billed {bytes_billed}, approx cost ${approx_cost}.",
        )

        if do_not_return_results:
            return True

        # job.result() blocks until the query has finished.
        result = job.result()
        if return_df:
            if result.total_rows and result.total_rows > 0:
                results_df = result.to_dataframe()
                return results_df
            return pd.DataFrame()
        return result


class BMProxy:
    """A proxy class for lazy initialization of the BM singleton."""

    _instance = None

    def __getattr__(self, name):
        """Forward attribute access to the BM instance, creating it if needed."""
        if BMProxy._instance is None:
            # Use the existing Singleton pattern from BM
            BMProxy._instance = BM()
        return getattr(BMProxy._instance, name)

    def __call__(self, *args, **kwargs):
        """Initialize or return the BM singleton."""
        if not args and not kwargs and BMProxy._instance is not None:
            # Just returning existing instance
            return BMProxy._instance

        # Create new instance with args/kwargs (or default)
        BMProxy._instance = BM(*args, **kwargs)
        return BMProxy._instance


# Create the global instance
bm = BM()

if __name__ == "__main__":
    pass
