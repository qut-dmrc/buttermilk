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
    TypeVar,
)

import coloredlogs
import google.cloud.logging  # Don't conflict with standard logging
import humanfriendly
import pandas as pd
import pydantic
import shortuuid
import weave
from langfuse import Langfuse
from weave.trace.weave_client import WeaveClient
from google.auth.credentials import Credentials as GoogleCredentials
from google.cloud import aiplatform, bigquery, storage
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from omegaconf import DictConfig
from pydantic import (
    PrivateAttr,
    model_validator,
)

from ._core.config import Project
from ._core.llms import LLMs
from ._core.log import logger
from .utils import save
from .utils.keys import SecretsManager
from .utils.utils import load_json_flexi

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
        return tuple((_convert_to_hashable_type(k), _convert_to_hashable_type(v)) for k, v in element.items())
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
    _gcp_project: str = PrivateAttr(default="")
    _gcp_credentials_cached: GoogleCredentials | None = PrivateAttr(default=None)  # Allow None initially
    _weave: WeaveClient = PrivateAttr()

    @property
    def weave(self) -> WeaveClient:
        if not hasattr(self, "_weave"):
            # Ensure run_info is available before accessing name/job
            if self.tracing and self.run_info and self.tracing.enabled and self.tracing.provider == "weave":
                collection = f"{self.run_info.name}-{self.run_info.job}"
                self._weave = weave.init(collection)
            else:
                raise RuntimeError("run_info/tracing details not set, cannot initialize Weave.")
        return self._weave

    @property
    def _gcp_credentials(self) -> GoogleCredentials:
        from google.auth import default
        from google.auth.transport.requests import Request  # Correct import needed here too

        if self._gcp_credentials_cached is None:
            billing_project = os.environ.get("google_billing_project", os.environ.get("GOOGLE_CLOUD_PROJECT", self._gcp_project))
            if not billing_project:
                self._gcp_credentials_cached, self._gcp_project = default()
                billing_project = self._gcp_project
                # raise ValueError("GOOGLE_CLOUD_PROJECT or google_billing_project environment variable not set.")

            # Use PrivateAttr default=None and check for None before using
            if self._gcp_credentials_cached is None:
                self._gcp_credentials_cached, self._gcp_project = default(
                    quota_project_id=billing_project,
                )

        # GCP tokens last 60 minutes and need to be refreshed after that
        # Ensure creds are not None before refreshing
        auth_request = Request()  # Use imported Request
        self._gcp_credentials_cached.refresh(auth_request)

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

    # @model_validator(mode="after") # Remove this validator
    # def ensure_config(self) -> Self:
    #     ... # Logic moved to setup_instance

    def setup_instance(self) -> None:
        """Performs setup requiring configuration (e.g., run_info, logger_cfg). Call after instantiation."""
        # Ensure run_info is set before proceeding with dependent setups
        if not self.run_info:
            # Try to get it from environment or raise error
            # This depends on how run_info is expected to be populated
            # For now, let's raise an error if it's critical
            raise ValueError("BM instance created without run_info, which is required for setup.")

        for cloud in self.clouds:
            if cloud.type == "gcp":
                # store authentication info
                os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get(
                    "GOOGLE_CLOUD_PROJECT",
                    cloud.project,  # type: ignore[attr-defined]
                )
                if hasattr(cloud, "quota_project_id"):  # Check attribute existence for safety
                    os.environ["google_billing_project"] = cloud.quota_project_id  # type: ignore[attr-defined]

                # authenticate here (triggers property getter)
                _ = self._gcp_credentials

            if cloud.type == "vertex":
                # initialize vertexai
                aiplatform.init(
                    project=cloud.project,  # type: ignore[attr-defined]
                    location=cloud.location,  # type: ignore[attr-defined]
                    staging_bucket=cloud.bucket,  # type: ignore[attr-defined]
                )
                # list available models
                # models = aiplatform.Model.list() # Potentially slow, consider if needed at init

        if self.logger_cfg:
            # Pass verbose safely using getattr
            self.setup_logging(verbose=getattr(self.logger_cfg, "verbose", False))

            # Print config to console and save to default save dir
            print(self.model_dump())
            try:
                self.save(
                    data=[self.model_dump(), self.run_info.model_dump()],
                    basename="config",
                    extension=".json",  # Default extension
                    save_dir=self.save_dir,
                )

            except Exception as e:
                logger.error(f"Could not save config to default save dir: {e}")

        # Ensure run_info exists before setting up tracing
        if self.tracing and self.run_info and self.tracing.enabled:
            collection = f"{self.run_info.name}-{self.run_info.job}"
            if self.tracing.provider == "traceloop":
                from traceloop.sdk import Traceloop

                Traceloop.init(
                    disable_batch=True,
                    api_key=self.tracing.api_key,
                )

            langfuse = Langfuse(
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"), 
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"), 
                host=os.getenv("LANGFUSE_HOST")
                )
            # elif self.tracing.provider == "promptflow":
            # from promptflow.tracing import start_trace

            # start_trace(
            #     resource_attributes={"run_id": self.run_info.run_id},
            #     collection=collection,
            # )

            pass

    @property
    def save_dir(self) -> str:
        if not self.run_info:
            # Raise error to ensure str return type for static analysis
            raise RuntimeError("run_info not set, cannot determine save_dir.")
        # run_info is guaranteed non-None here
        return self.run_info.save_dir

    def save(self, data, save_dir=None, extension: str | None = None, **kwargs):
        """Failsafe save method."""
        save_dir = save_dir or self.save_dir
        # Provide default extension if None
        effective_extension = extension if extension is not None else ".json"
        result = save.save(data=data, save_dir=save_dir, extension=effective_extension, **kwargs)
        logger.info(
            dict(
                message=f"Saved data to: {result}",
                uri=result,
                run_id=self.run_info.run_id if self.run_info else "unknown",
            ),
        )
        return result

    def setup_logging(
        self,
        verbose=False,
    ) -> None:
        # Ensure run_info is available before setting up GCP logging
        if self.logger_cfg and self.logger_cfg.type == "gcp":
            if not self.run_info:
                raise RuntimeError("run_info must be set before configuring GCP logging.")
            if not hasattr(self.logger_cfg, "project"):
                raise RuntimeError("GCP logger config missing 'project' attribute.")
            if not hasattr(self.logger_cfg, "location"):
                raise RuntimeError("GCP logger config missing 'location' attribute.")

        root_logger = logging.getLogger()

        # Remove existing handlers from the root logger to avoid duplicates
        # or conflicts if setup_logging is called multiple times or by other libraries.
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # for clarity
        from ._core.log import logger as bm_logger

        console_format = "%(asctime)s %(hostname)s %(name)s %(filename)s:%(lineno)d %(levelname)s %(message)s"
        coloredlogs.install(
            logger=bm_logger,
            fmt=console_format,
            isatty=True,
            stream=sys.stderr,
            level=logging.DEBUG if verbose else logging.INFO
        )

        resource = None
        if self.logger_cfg and self.logger_cfg.type == "gcp":
            # Labels for cloud logger
            # We've already checked run_info and attributes exist above
            resource = google.cloud.logging.Resource(
                type="generic_task",
                labels={
                    "project_id": self.logger_cfg.project,  # type: ignore[attr-defined]
                    "location": self.logger_cfg.location,  # type: ignore[attr-defined]
                    "namespace": self.run_info.name,  # type: ignore[union-attr] # Checked above
                    "job": self.run_info.job,  # type: ignore[union-attr] # Checked above
                    "task_id": self.run_info.run_id,  # type: ignore[union-attr] # Checked above
                },
            )

            cloudHandler = CloudLoggingHandler(
                client=self.gcs_log_client,  # Access via property
                resource=resource,
                name=self.run_info.name,  # type: ignore[union-attr] # Checked above
                labels=self.run_info.model_dump(  # type: ignore[union-attr] # Checked above
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

        message = f"Logging set up for: {self.run_info}. Ready. Save dir: {self.save_dir}"

        if resource:
            message = f"{message} {resource}"  # Append resource info if available

        bm_logger.info(message, extra=dict(run=self.run_info.model_dump() if self.run_info else {}))

        try:
            from importlib.metadata import version

            bm_logger.debug(f"Buttermilk version is: {version('buttermilk')}")
        except:
            pass

    @property
    def gcs_log_client(self) -> google.cloud.logging.Client:
        # Ensure logger_cfg and project exist before creating client
        if not self.logger_cfg or not hasattr(self.logger_cfg, "project"):
            raise RuntimeError("Logger config with GCP project needed for GCS Log Client.")

        if _REGISTRY.get("gcslogging") is None:
            _REGISTRY["gcslogging"] = google.cloud.logging.Client(
                project=self.logger_cfg.project,
                credentials=self._gcp_credentials_cached,  # type: ignore[attr-defined]  # Pass credentials
            )
        return _REGISTRY["gcslogging"]

    @property
    def gcs(self) -> storage.Client:
        # Ensure _gcp_project is set (happens in _gcp_credentials getter)
        if not hasattr(self, "_gcp_project"):
            _ = self._gcp_credentials  # Trigger credential loading if not done yet
            if not hasattr(self, "_gcp_project"):  # Check again
                raise RuntimeError("GCP project not determined. Ensure GCP cloud config is present.")

        if _REGISTRY.get("gcs") is None:
            _REGISTRY["gcs"] = storage.Client(project=self._gcp_project, credentials=self._gcp_credentials_cached)  # Pass credentials
        return _REGISTRY["gcs"]

    @property
    def secret_manager(self) -> SecretsManager:
        if not self.secret_provider:
            raise RuntimeError("Secret provider configuration is missing.")

        if _REGISTRY.get("secret_manager") is None:
            _REGISTRY["secret_manager"] = SecretsManager(
                **self.secret_provider.model_dump(),
            )
        return _REGISTRY["secret_manager"]

    @property
    def credentials(self) -> dict[str, str]:
        # Ensure secret_manager is available
        secret_mgr = self.secret_manager  # Trigger secret_manager init if needed

        if _REGISTRY.get("credentials") is None:
            creds = secret_mgr.get_secret(
                cfg_key=_SHARED_CREDENTIALS_KEY,
            )
            # Ensure the returned value is a dictionary
            if not isinstance(creds, dict):
                raise TypeError(f"Expected credentials secret to be a dict, got {type(creds)}")
            _REGISTRY["credentials"] = creds
        # Return type assertion for clarity
        return _REGISTRY["credentials"]

    @property
    def llms(self) -> LLMs:
        # Ensure secret_manager is available
        secret_mgr = self.secret_manager  # Trigger secret_manager init if needed

        if _REGISTRY.get("llms") is None:
            connections = None
            try:
                # Blocking IO
                contents = Path(CONFIG_CACHE_PATH).read_text(encoding="utf-8")
                connections = load_json_flexi(contents)
            except Exception:
                # Blocking call
                connections = secret_mgr.get_secret(
                    cfg_key=_MODELS_CFG_KEY,
                )
                try:
                    # Blocking IO
                    Path(CONFIG_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
                    Path(CONFIG_CACHE_PATH).write_text(
                        json.dumps(connections),
                        encoding="utf-8",
                    )
                except Exception as e:
                    logger.error(f"Unable to cache connections: {e}, {e.args}")

            # Ensure connections is a dictionary before passing to LLMs
            if not isinstance(connections, dict):
                raise TypeError(f"LLM connections loaded from secret/cache is not a dict: {type(connections)}")

            _REGISTRY["llms"] = LLMs(connections=connections)
        # Return type assertion for clarity
        return _REGISTRY["llms"]

    @property
    def bq(self) -> bigquery.Client:
        # Ensure _gcp_project is set (happens in _gcp_credentials getter)
        if not hasattr(self, "_gcp_project"):
            _ = self._gcp_credentials  # Trigger credential loading if not done yet
            if not hasattr(self, "_gcp_project"):  # Check again
                raise RuntimeError("GCP project not determined. Ensure GCP cloud config is present.")

        if _REGISTRY.get("bq") is None:
            _REGISTRY["bq"] = bigquery.Client(project=self._gcp_project, credentials=self._gcp_credentials_cached)  # Pass credentials
        return _REGISTRY["bq"]

    def run_query(
        self,
        sql,
        destination=None,
        overwrite=False,
        do_not_return_results=False,
        save_to_gcs=False,
        return_df=True,
    ) -> pd.DataFrame | Any | bool:  # Allow Any for non-df results, bool for do_not_return
        t0 = datetime.datetime.now()

        job_config_dict = {
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
            job_config_dict["destination"] = destination
            job_config_dict["allow_large_results"] = True

            # Set write_disposition directly on the config object
            # job_config_dict["write_disposition"] = "WRITE_TRUNCATE" if overwrite else "WRITE_APPEND"
            pass  # Will set on object below

        job_config = bigquery.QueryJobConfig(**job_config_dict)
        # Set attributes directly
        if destination:
            job_config.destination = destination
            job_config.allow_large_results = True
            job_config.write_disposition = "WRITE_TRUNCATE" if overwrite else "WRITE_APPEND"

        job = self.bq.query(sql, job_config=job_config)

        bytes_billed = job.total_bytes_billed
        cache_hit = job.cache_hit

        if bytes_billed:
            approx_cost = bytes_billed * GOOGLE_BQ_PRICE_PER_BYTE
            bytes_billed_str = humanfriendly.format_size(bytes_billed)  # Use different var name
            approx_cost_str = humanfriendly.format_number(approx_cost)  # Use different var name
        else:
            bytes_billed_str = "N/A"  # Assign to string var
            approx_cost_str = "unknown"  # Assign to string var
        time_taken = datetime.datetime.now() - t0
        logger.info(
            f"Query stats: Ran in {time_taken} seconds, cache hit: {cache_hit}, billed {bytes_billed_str}, approx cost ${approx_cost_str}.",
        )

        if do_not_return_results:
            return True  # Return bool as per type hint

        # job.result() blocks until the query has finished.
        result = job.result()
        if return_df:
            if result.total_rows and result.total_rows > 0:
                results_df = result.to_dataframe()
                return results_df
            return pd.DataFrame()
        return result  # Return Any (BigQuery result object)


# Create the global instance directly using the BM class
bm = BM()

if __name__ == "__main__":
    pass
