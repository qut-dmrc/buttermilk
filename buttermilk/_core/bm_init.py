# Configurations are stored in yaml files and managed by the Hydra library.
#
# Projects will  have a common config.yaml file that will be used to store configurations that
# are common to all the experiments in the project. Individual experiments will have their own
# config.yaml file that will be used to store configurations that are specific to that experiment.
# Authentication credentials are stored in secure cloud key/secret vaults on GCP, Azure, or AWS.
# The configuration files will be used to store the paths to the authentication credentials in
# the cloud vaults.

from __future__ import annotations  # Enable postponed annotations

import asyncio
import base64
import datetime
import json
import logging
import os
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import mkdtemp
from typing import (
    Any,
    Self,
)

import coloredlogs
import google.cloud.logging  # Don't conflict with standard logging
import humanfriendly
import pandas as pd
import psutil
import pydantic
import shortuuid
import weave
from cloudpathlib import AnyPath, CloudPath
from google.auth.credentials import Credentials as GoogleCredentials
from google.cloud import bigquery, storage
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from omegaconf import DictConfig
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
)
from rich import print
from vertexai import init as aiplatform_init
from weave.trace.weave_client import WeaveClient

from buttermilk._core.config import CloudProviderCfg, DataSourceConfig, Tracing
from buttermilk._core.keys import SecretsManager
from buttermilk._core.llms import LLMs
from buttermilk._core.log import logger
from buttermilk.utils import save
from buttermilk.utils.utils import load_json_flexi

CONFIG_CACHE_PATH = ".cache/buttermilk/models.json"
_MODELS_CFG_KEY = "models_secret"
_SHARED_CREDENTIALS_KEY = "credentials_secret"

# https://cloud.google.com/bigquery/pricing
GOOGLE_BQ_PRICE_PER_BYTE = 5 / 10e12  # $5 per tb.


_global_run_id = ""


def _make_run_id() -> str:
    global _global_run_id
    if _global_run_id:
        return _global_run_id
    # Create a unique identifier for this run
    node_name = platform.uname().node
    username = psutil.Process().username()
    # get rid of windows domain if present
    username = str.split(username, "\\")[-1]

    # The ISO 8601 format has too many special characters for a filename, so we'll use a simpler format
    run_time = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%MZ")

    run_id = f"{run_time}-{shortuuid.uuid()[:4]}-{node_name}-{username}"
    _global_run_id = run_id
    return run_id


class SessionInfo(BaseModel):
    """This is a model purely for exporting configuration information."""

    platform: str = "local"
    name: str
    job: str
    run_id: str = Field(default_factory=_make_run_id)
    max_concurrency: int = -1
    ip: str | None = Field(default=None)
    node_name: str = Field(default_factory=lambda: platform.uname().node)
    save_dir: str | None = None
    flow_api: str | None = None

    _get_ip_task: asyncio.Task | None = PrivateAttr(default=None)  # Initialize as None
    _ip: str | None = PrivateAttr(default=None)

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=False)


# Inherit from BaseModel and the simplified Singleton
class BM(SessionInfo):
    connections: Sequence[str] = Field(default_factory=list)
    secret_provider: CloudProviderCfg = Field(default=None)
    logger_cfg: CloudProviderCfg = Field(default=None)
    pubsub: CloudProviderCfg = Field(default=None)
    clouds: list[CloudProviderCfg] = Field(default_factory=list)
    tracing: Tracing | None = Field(default_factory=Tracing)
    datasets: dict[str, DataSourceConfig] = Field(default_factory=dict)

    save_dir_base: str = Field(
        default_factory=mkdtemp,
        validate_default=True,
    )  # Default to temp dir

    _gcp_project: str = PrivateAttr(default="")
    _gcp_credentials_cached: GoogleCredentials | None = PrivateAttr(
        default=None,
    )  # Allow None initially
    _weave: WeaveClient | None = PrivateAttr(default=None)  # Initialize as None

    _tracer: trace.Tracer = PrivateAttr()

    # Private attributes for caching client instances
    _gcs_log_client_cached: google.cloud.logging.Client | None = PrivateAttr(default=None)
    _gcs_cached: storage.Client | None = PrivateAttr(default=None)
    _secret_manager_cached: SecretsManager | None = PrivateAttr(default=None)
    _credentials_cached: dict[str, str] | None = PrivateAttr(default=None)
    _llms_cached: LLMs | None = PrivateAttr(default=None)
    _bq_cached: bigquery.Client | None = PrivateAttr(default=None)

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow arbitrary types for things like GoogleCredentials, WeaveClient, etc.
        populate_by_name=True,
        exclude_none=True,
        exclude_unset=True,
    )

    @pydantic.field_validator("save_dir_base", mode="before")
    def get_save_dir(cls, save_dir_base, values) -> str:
        if isinstance(save_dir_base, str):
            pass
        elif isinstance(save_dir_base, Path):
            save_dir_base = save_dir_base.as_posix()
        elif isinstance(save_dir_base, CloudPath):
            save_dir_base = save_dir_base.as_uri()
        else:
            raise ValueError(
                f"save_dir_base must be a string, Path, or CloudPath, got {type(save_dir_base)}",
            )
        return save_dir_base

    @pydantic.model_validator(mode="before")
    @classmethod
    def _remove_target(cls, values: dict[str, Any]) -> dict[str, Any]:
        # Remove the hydra target attribute from the values dictionary
        values.pop("_target_", None)
        return values  # Return the dict directly, not a new instance

    @pydantic.model_validator(mode="after")
    def set_full_save_dir(self) -> Self:
        # This validator runs *after* BaseModel initialization.
        save_dir = AnyPath(self.save_dir_base) / self.name / self.job / self.run_id
        self.save_dir = str(save_dir)
        return self

    @property
    def weave(self) -> WeaveClient:
        # Initialize weave only if it hasn't been initialized yet
        if self._weave is None:
            collection = f"{self.name}-{self.job}"
            try:
                # Ensure credentials are loaded before accessing WANDB_API_KEY
                _ = self.credentials  # Trigger credential loading
                self._weave = weave.init(collection)
                logger.info(f"Weave tracing initialized for collection: {collection}")
            except Exception as e:
                # Crash out if Weave initialization fails
                logger.error(f"Failed to initialize Weave tracing: {e}")
                raise RuntimeError(f"Failed to initialize Weave tracing: {e}") from e

        return self._weave

    @property
    def run_info(self) -> SessionInfo:
        # Access private attributes directly
        return SessionInfo(
            platform=self.platform,
            name=self.name,
            job=self.job,
            node_name=self.node_name,
            ip=self._ip,
            save_dir=self.save_dir,
            flow_api=self.flow_api,
        )

    @property
    def _gcp_credentials(self) -> GoogleCredentials:
        from google.auth import default
        from google.auth.transport.requests import Request

        if self._gcp_credentials_cached is None:

            # Find GCP cloud config
            gcp_cloud_cfg = next((c for c in self.clouds if c and hasattr(c, "type") and c.type == "gcp"), None)

            if gcp_cloud_cfg:
                # Use project and quota_project_id from config if available
                project_id = getattr(gcp_cloud_cfg, "project", None)
                quota_project_id = getattr(gcp_cloud_cfg, "quota_project_id", project_id)

                os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get("GOOGLE_CLOUD_PROJECT", project_id)
                os.environ["google_billing_project"] = os.environ.get("google_billing_project", quota_project_id)

                scopes = ["https://www.googleapis.com/auth/cloud-platform"]
                self._gcp_credentials_cached, self._gcp_project = default(
                    quota_project_id=quota_project_id,
                    scopes=scopes,
                )

        # GCP tokens last 60 minutes and need to be refreshed after that
        auth_request = Request()  # Use imported Request
        # Check if credentials support refresh before calling
        if hasattr(self._gcp_credentials_cached, "refresh"):
            self._gcp_credentials_cached.refresh(auth_request)

        return self._gcp_credentials_cached

    @property
    def logger(self) -> logging.Logger:
        return logger

    def setup_instance(self, cfg: DictConfig = None) -> None:
        """Initializes the BM singleton instance after validation."""
        # This method is called *after* the BaseModel initialization is complete
        # and the instance attributes are set according to the config.

        from buttermilk.utils import get_ip

        # Initialize IP task asynchronously if needed
        # Check if an event loop is running before creating a task
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                if not hasattr(self, "_get_ip_task") or self._get_ip_task is None:
                    self._get_ip_task = asyncio.create_task(get_ip())
            else:
                # No event loop running, skip IP task creation
                pass
        except RuntimeError:
            # No current event loop
            pass

        # Only process clouds if we have them and they're properly initialized
        if hasattr(self, "clouds") and self.clouds:
            for cloud in self.clouds:
                if not cloud or not hasattr(cloud, "type"):
                    continue  # Skip invalid cloud entries

                if cloud.type == "gcp":
                    # store authentication info - this is now handled in _gcp_credentials property
                    pass  # Keep this block for clarity if needed later

                if cloud.type == "vertex":
                    # initialize vertexai
                    # Ensure project, location, bucket attributes exist before accessing
                    project = getattr(cloud, "project", None)
                    location = getattr(cloud, "location", None)
                    bucket = getattr(cloud, "bucket", None)
                    if project and location and bucket:
                        try:
                            aiplatform_init(
                                project=project,
                                location=location,
                                staging_bucket=bucket,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to initialize Vertex AI: {e}")
                    else:
                        logger.warning(
                            "Skipping Vertex AI initialization due to missing project, location, or bucket in config.",
                        )

        if self.logger_cfg:
            # Pass verbose safely using getattr
            self.setup_logging(verbose=getattr(self.logger_cfg, "verbose", False))

            # Print config to console and save to default save dir
            print(self.model_dump())
            try:
                # Ensure save_dir is set before saving
                if self.save_dir:
                    self.save(
                        data=[self.model_dump(), self.run_info.model_dump()],
                        basename="config",
                        extension=".json",  # Default extension
                        save_dir=self.save_dir,
                    )
                else:
                    logger.warning("save_dir is not set. Skipping saving config.")

            except Exception as e:
                logger.error(f"Could not save config to default save dir: {e}")

        # Tracing setup (OTEL)
        WANDB_BASE_URL = "https://trace.wandb.ai"
        OTEL_EXPORTER_OTLP_ENDPOINT = f"{WANDB_BASE_URL}/otel/v1/traces"
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = OTEL_EXPORTER_OTLP_ENDPOINT
        os.environ["TRACELOOP_BASE_URL"] = OTEL_EXPORTER_OTLP_ENDPOINT
        # WANDB_API_KEY: get from https://wandb.ai/authorize
        # Ensure credentials are available before accessing WANDB_API_KEY
        # Accessing self.credentials will trigger loading if not cached
        try:
            creds = self.credentials
            if creds and "WANDB_API_KEY" in creds and "WANDB_PROJECT" in creds:
                AUTH = base64.b64encode(f"api:{creds['WANDB_API_KEY']}".encode()).decode()

                OTEL_EXPORTER_OTLP_HEADERS = {
                    "Authorization": f"Basic {AUTH}",
                    "project_id": creds["WANDB_PROJECT"],
                }

                # Initialize the OpenTelemetry SDK
                from opentelemetry.sdk import trace as trace_sdk

                tracer_provider = trace_sdk.TracerProvider()

                # Configure the OTLP exporter
                exporter = OTLPSpanExporter(
                    endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
                    headers=OTEL_EXPORTER_OTLP_HEADERS,
                )

                # Add the exporter to the tracer provider
                tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

                # Set the global tracer provider
                trace.set_tracer_provider(tracer_provider)
                self._tracer = trace.get_tracer(__name__)  # Get a tracer instance

                logger.info(
                    "Tracing setup configured (OTEL). Weave initialization happens on first access.",
                )
            else:
                logger.warning("Wandb credentials missing. OpenTelemetry tracing not fully configured.")
        except Exception as e:
            logger.warning(f"Error during OpenTelemetry tracing setup: {e}. Continuing without OTEL tracing.")

    def save(self, data, save_dir=None, extension: str | None = None, **kwargs):
        """Failsafe save method."""
        save_dir = save_dir or self.save_dir
        # Provide default extension if None
        effective_extension = extension if extension is not None else ".json"
        try:
            result = save.save(
                data=data,
                save_dir=save_dir,
                extension=effective_extension,
                **kwargs,
            )
            logger.info(
                dict(
                    message=f"Saved data to: {result}",
                    uri=result,
                    run_id=self.run_id,
                ),
            )
            return result
        except Exception as e:
            logger.error(f"Failed to save data to {save_dir}: {e}")
            return None  # Indicate save failure

    def setup_logging(
        self,
        verbose=False,
    ) -> None:
        if self.logger_cfg and self.logger_cfg.type == "gcp":
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
        from .log import logger as bm_logger

        console_format = "%(asctime)s %(hostname)s %(name)s %(filename)s:%(lineno)d %(levelname)s %(message)s"
        coloredlogs.install(
            logger=bm_logger,
            fmt=console_format,
            isatty=True,
            stream=sys.stderr,
            level=logging.DEBUG if verbose else logging.INFO,
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
                    "namespace": self.name,  # type: ignore[union-attr] # Checked above
                    "job": self.job,  # type: ignore[union-attr] # Checked above
                    "task_id": self.run_id,  # type: ignore[union-attr] # Checked above # Access private attribute
                },
            )

            cloudHandler = CloudLoggingHandler(
                client=self.gcs_log_client,  # Access via property
                resource=resource,
                name=self.name,  # type: ignore[union-attr] # Checked above
                labels=self.model_dump(  # type: ignore[union-attr] # Checked above
                    include={"run_id", "name", "job", "platform", "username"},
                ),
            )
            cloudHandler.setLevel(
                logging.INFO,
            )  # Cloud logging never uses the DEBUG level, there's just too much data. Print debug to console only.
            bm_logger.addHandler(cloudHandler)

        # --- Set logging level: warning for others, either debug or info for us ---
        root_logger.setLevel(logging.WARNING)
        # Iterate over a copy of the keys as loggerDict can change during iteration
        for logger_str in list(logging.Logger.manager.loggerDict.keys()):
            try:
                # Avoid setting level on loggers that might not be fully initialized or are placeholders
                if isinstance(logging.Logger.manager.loggerDict[logger_str], logging.Logger):
                    logging.getLogger(logger_str).setLevel(logging.WARNING)
            except Exception:
                # Handle potential errors during logger access
                pass

        if verbose:
            bm_logger.setLevel(logging.DEBUG)
        else:
            # Check if an event loop is running before setting debug mode
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.set_debug(False)
                else:
                    # If no loop is running, setting debug might not be relevant or could raise error
                    pass  # Or handle appropriately
            except RuntimeError:
                # No current event loop
                pass

            bm_logger.setLevel(logging.INFO)

        message = (
            f"Logging set up for: {self.run_info}. Ready. Save dir: {self.save_dir}"
        )

        if resource:
            message = f"{message} {resource}"  # Append resource info if available

        bm_logger.info(
            message,
            extra=dict(run=self.run_info.model_dump() if self.run_info else {}),
        )

        try:
            from importlib.metadata import version

            bm_logger.debug(f"Buttermilk version is: {version('buttermilk')}")
        except:
            pass

    @property
    def gcs_log_client(self) -> google.cloud.logging.Client:
        # Ensure logger_cfg and project exist before creating client
        if not self.logger_cfg or not hasattr(self.logger_cfg, "project"):
            # Attempt to get config from registry if available (e.g., in tests)
            cfg = ConfigRegistry.get_config()
            if cfg and hasattr(cfg, "bm") and hasattr(cfg.bm, "logger_cfg") and hasattr(cfg.bm.logger_cfg, "project"):
                self.logger_cfg = cfg.bm.logger_cfg  # type: ignore
            else:
                raise RuntimeError(
                    "Logger config with GCP project needed for GCS Log Client.",
                )

        if self._gcs_log_client_cached is None:
            # Ensure credentials are loaded before creating client
            _ = self._gcp_credentials  # Trigger credential loading
            self._gcs_log_client_cached = google.cloud.logging.Client(
                project=self.logger_cfg.project,  # type: ignore[attr-defined]
                credentials=self._gcp_credentials_cached,  # type: ignore[attr-defined]  # Pass credentials
            )
        return self._gcs_log_client_cached

    @property
    def gcs(self) -> storage.Client:
        # Ensure _gcp_project is set (happens in _gcp_credentials getter)
        if not hasattr(self, "_gcp_project") or not self._gcp_project:
            _ = self._gcp_credentials  # Trigger credential loading if not done yet
            if not hasattr(self, "_gcp_project") or not self._gcp_project:
                raise RuntimeError(
                    "GCP project not determined. Ensure GCP cloud config is present or GOOGLE_CLOUD_PROJECT env var is set.",
                )

        if self._gcs_cached is None:
            # Ensure credentials are loaded before creating client
            _ = self._gcp_credentials  # Trigger credential loading
            self._gcs_cached = storage.Client(
                project=self._gcp_project,
                credentials=self._gcp_credentials_cached,
            )  # Pass credentials
        return self._gcs_cached

    @property
    def secret_manager(self) -> SecretsManager:
        if self._secret_manager_cached is None:
            if not self.secret_provider:
                # Attempt to get config from registry if available (e.g., in tests)
                cfg = ConfigRegistry.get_config()
                if cfg and hasattr(cfg, "bm") and hasattr(cfg.bm, "secret_provider"):
                    self.secret_provider = cfg.bm.secret_provider  # type: ignore
                else:
                    raise RuntimeError("Secret provider configuration is missing.")

            self._secret_manager_cached = SecretsManager(
                **self.secret_provider.model_dump(),  # type: ignore[union-attr]
            )
        return self._secret_manager_cached

    @property
    def credentials(self) -> dict[str, str]:
        # Ensure secret_manager is available
        secret_mgr = self.secret_manager  # Trigger secret_manager init if needed

        if self._credentials_cached is None:
            creds = secret_mgr.get_secret(
                cfg_key=_SHARED_CREDENTIALS_KEY,
            )
            # Ensure the returned value is a dictionary
            if not isinstance(creds, dict):
                raise TypeError(
                    f"Expected credentials secret to be a dict, got {type(creds)}",
                )
            self._credentials_cached = creds
        # Return type assertion for clarity
        return self._credentials_cached

    @property
    def llms(self) -> LLMs:
        if self._llms_cached is None:
            connections = None
            try:
                # Blocking IO
                contents = Path(CONFIG_CACHE_PATH).read_text(encoding="utf-8")
                connections = load_json_flexi(contents)
            except Exception:
                # Blocking call
                # Ensure secret_manager is available
                try:
                    connections = self.secret_manager.get_secret(
                        cfg_key=_MODELS_CFG_KEY,
                    )
                except Exception as e:
                    logger.error(f"Failed to load LLM connections from secret manager: {e}")
                    raise RuntimeError("Failed to load LLM connections.") from e

                try:
                    # Optionally cache the fetched connections to a file for next time
                    Path(CONFIG_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
                    Path(CONFIG_CACHE_PATH).write_text(json.dumps(connections), encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Could not cache LLM connections to {CONFIG_CACHE_PATH}: {e}")

            # Ensure connections is a dictionary before passing to LLMs
            if not isinstance(connections, dict):
                raise TypeError(
                    f"LLM connections loaded from secret/cache is not a dict: {type(connections)}",
                )

            self._llms_cached = LLMs(connections=connections)
        # Return type assertion for clarity
        return self._llms_cached

    @property
    def bq(self) -> bigquery.Client:
        # Ensure _gcp_project is set (happens in _gcp_credentials getter)
        if not hasattr(self, "_gcp_project") or not self._gcp_project:
            _ = self._gcp_credentials  # Trigger credential loading if not done yet
            if not hasattr(self, "_gcp_project") or not self._gcp_project:
                raise RuntimeError(
                    "GCP project not determined. Ensure GCP cloud config is present or GOOGLE_CLOUD_PROJECT env var is set.",
                )

        if self._bq_cached is None:
            # Ensure credentials are loaded before creating client
            _ = self._gcp_credentials  # Trigger credential loading
            self._bq_cached = bigquery.Client(
                project=self._gcp_project,
                credentials=self._gcp_credentials_cached,
            )
        return self._bq_cached

    def run_query(
        self,
        sql,
        destination=None,
        overwrite=False,
        do_not_return_results=False,
        save_to_gcs=False,
        return_df=True,
    ) -> (
        pd.DataFrame | Any | bool
    ):  # Allow Any for non-df results, bool for do_not_return
        t0 = datetime.datetime.now()

        job_config_dict = {
            "use_legacy_sql": False,
        }

        # Cannot set write_disposition if saving to GCS
        if save_to_gcs:
            # Tell BigQuery to save the results to a specific GCS location
            # Ensure save_dir is set before using it
            if not self.save_dir:
                logger.error("save_dir is not set. Cannot save query results to GCS.")
                return False  # Indicate failure

            gcs_results_uri = f"{self.save_dir}/query_{shortuuid.uuid()}/*.json"
            job_config_dict["destination_uris"] = [gcs_results_uri]
            job_config_dict["write_disposition"] = bigquery.WriteDisposition.WRITE_TRUNCATE  # Overwrite existing files

        elif destination:
            # Configure destination table if specified
            job_config_dict["destination"] = destination
            job_config_dict["write_disposition"] = (
                bigquery.WriteDisposition.WRITE_TRUNCATE
                if overwrite
                else bigquery.WriteDisposition.WRITE_EMPTY
            )

        job_config = bigquery.QueryJobConfig(**job_config_dict)
        # Set attributes directly
        try:
            job = self.bq.query(sql, job_config=job_config)
        except Exception as e:
            logger.error(f"BigQuery query failed: {e}")
            return False  # Indicate failure

        bytes_billed = job.total_bytes_billed
        cache_hit = job.cache_hit

        if bytes_billed:
            approx_cost = bytes_billed * GOOGLE_BQ_PRICE_PER_BYTE
            bytes_billed_str = humanfriendly.format_size(
                bytes_billed,
            )  # Use different var name
            approx_cost_str = humanfriendly.format_number(
                approx_cost,
            )  # Use different var name
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
        try:
            result = job.result()
            if return_df:
                if result.total_rows and result.total_rows > 0:
                    results_df = result.to_dataframe()
                    return results_df
                return pd.DataFrame()  # Return empty DataFrame if no rows
            return result  # Return Any (BigQuery result object)
        except Exception as e:
            logger.error(f"Failed to get BigQuery query results: {e}")
            return False  # Indicate failure


if __name__ == "__main__":
    pass
