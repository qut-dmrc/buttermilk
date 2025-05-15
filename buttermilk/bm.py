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
from buttermilk._core.singleton import Singleton
from buttermilk._core.types import SessionInfo
from buttermilk.utils import get_ip, save
from buttermilk.utils.utils import load_json_flexi

CONFIG_CACHE_PATH = ".cache/buttermilk/models.json"
_MODELS_CFG_KEY = "models_secret"
_SHARED_CREDENTIALS_KEY = "credentials_secret"

# https://cloud.google.com/bigquery/pricing
GOOGLE_BQ_PRICE_PER_BYTE = 5 / 10e12  # $5 per tb.


class SessionInfo(BaseModel):
    platform: str = "local"
    name: str
    job: str
    run_id: str
    max_concurrency: int = -1
    ip: str = Field(default="")
    node_name: str = Field(default_factory=lambda: platform.uname().node)
    username: str = Field(
        default_factory=lambda: psutil.Process().username().split("\\")[-1],
    )
    save_dir: str | None = None
    flow_api: str | None = None

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=False)


class BM(Singleton, BaseModel):

    connections: Sequence[str] = Field(default_factory=list)
    secret_provider: CloudProviderCfg = Field(default=None)
    logger_cfg: CloudProviderCfg = Field(default=None)
    pubsub: CloudProviderCfg = Field(default=None)
    clouds: list[CloudProviderCfg] = Field(default_factory=list)
    tracing: Tracing | None = Field(default_factory=Tracing)
    run_info: SessionInfo | None = Field(
        default=None,
        description="Information about the context in which this project runs",
    )
    datasets: dict[str, DataSourceConfig] = Field(default_factory=dict)

    platform: str = "local"
    name: str
    job: str
    run_id: str
    max_concurrency: int = -1
    node_name: str = Field(default_factory=lambda: platform.uname().node)
    username: str = Field(
        default_factory=lambda: psutil.Process().username().split("\\")[-1],
    )
    save_dir: str | None = None
    flow_api: str | None = None
    _get_ip_task: asyncio.Task
    _ip: str = PrivateAttr()

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
        arbitrary_types_allowed=False,
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

    @pydantic.model_validator(mode="after")
    def set_full_save_dir(self) -> Self:
        save_dir = AnyPath(self.save_dir_base) / self.name / self.job / self.run_id
        self.save_dir = str(save_dir)
        return self

    @property
    def weave(self) -> WeaveClient:
        # Initialize weave only if it hasn't been initialized yet
        if self._weave is None:
            collection = f"{self.name}-{self.job}"
            try:
                self._weave = weave.init(collection)
                logger.info(f"Weave tracing initialized for collection: {collection}")
            except Exception as e:
                # Crash out if Weave initialization fails
                logger.error(f"Failed to initialize Weave tracing: {e}")
                raise RuntimeError(f"Failed to initialize Weave tracing: {e}") from e

        return self._weave

    async def get_ip(self):
        if not self.ip:
            self.ip = await get_ip()

    @property
    def run_info(self) -> SessionInfo:
        return SessionInfo(platform=self.platform, name=self.name, job=self.job, run_id=self.run_id,
                           node_name=self.node_name, username=self.username, ip=self.ip, save_dir=self.save_dir,
                           flow_api=self.flow_api)

    @property
    def _gcp_credentials(self) -> GoogleCredentials:
        from google.auth import default
        from google.auth.transport.requests import Request

        if self._gcp_credentials_cached is None:
            billing_project = os.environ.get(
                "google_billing_project",
                os.environ.get("GOOGLE_CLOUD_PROJECT", self._gcp_project),
            )
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            if not billing_project:
                self._gcp_credentials_cached, self._gcp_project = default(scopes=scopes)
                billing_project = self._gcp_project
            else:
                self._gcp_credentials_cached, self._gcp_project = default(
                    quota_project_id=billing_project,
                    scopes=scopes,
                )

        # GCP tokens last 60 minutes and need to be refreshed after that
        auth_request = Request()  # Use imported Request
        # Check if credentials support refresh before calling
        if hasattr(self._gcp_credentials_cached, "refresh"):
            self._gcp_credentials_cached.refresh(auth_request)

        return self._gcp_credentials_cached

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",  # Allow extra fields
    )

    @property
    def logger(self) -> logging.Logger:
        return logger

    # Removed model_validator(mode="before") and get_vars

    # Renamed from setup_instance and removed model_validator
    def __init__(self, **data: Any) -> None:
        """Initializes the BM singleton instance after validation."""
        super().__init__(**data)  # Call BaseModel's __init__ to populate fields

        for cloud in self.clouds:
            if cloud.type == "gcp":
                # store authentication info
                os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get(
                    "GOOGLE_CLOUD_PROJECT",
                    cloud.project,  # type: ignore[attr-defined]
                )
                if hasattr(
                    cloud, "quota_project_id",
                ):  # Check attribute existence for safety
                    os.environ["google_billing_project"] = cloud.quota_project_id  # type: ignore[attr-defined]

                # authenticate here (triggers property getter)
                _ = self._gcp_credentials

            if cloud.type == "vertex":
                # initialize vertexai
                aiplatform_init(
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
        from opentelemetry.sdk import trace as trace_sdk

        WANDB_BASE_URL = "https://trace.wandb.ai"
        OTEL_EXPORTER_OTLP_ENDPOINT = f"{WANDB_BASE_URL}/otel/v1/traces"
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = OTEL_EXPORTER_OTLP_ENDPOINT
        os.environ["TRACELOOP_BASE_URL"] = OTEL_EXPORTER_OTLP_ENDPOINT
        # WANDB_API_KEY: get from https://wandb.ai/authorize
        # Ensure credentials are available before accessing WANDB_API_KEY
        if self.credentials and "WANDB_API_KEY" in self.credentials and "WANDB_PROJECT" in self.credentials:
            AUTH = base64.b64encode(f"api:{self.credentials['WANDB_API_KEY']}".encode()).decode()

            OTEL_EXPORTER_OTLP_HEADERS = {
                "Authorization": f"Basic {AUTH}",
                "project_id": self.credentials["WANDB_PROJECT"],
            }

            # Initialize the OpenTelemetry SDK
            tracer_provider = trace_sdk.TracerProvider()

            # Configure the OTLP exporter
            exporter = OTLPSpanExporter(
                endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
                headers=OTEL_EXPORTER_OTLP_HEADERS,
            )

            # Add the exporter to the tracer provider
            tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

            logger.info(
                "Tracing setup configured (OTEL). Weave initialization happens on first access.",
            )
        else:
            logger.warning("Wandb credentials missing. OpenTelemetry tracing not fully configured.")

    def save(self, data, save_dir=None, extension: str | None = None, **kwargs):
        """Failsafe save method."""
        save_dir = save_dir or self.save_dir
        # Provide default extension if None
        effective_extension = extension if extension is not None else ".json"
        result = save.save(
            data=data, save_dir=save_dir, extension=effective_extension, **kwargs,
        )
        logger.info(
            dict(
                message=f"Saved data to: {result}",
                uri=result,
                run_id=self.run_id,
            ),
        )
        return result

    def setup_logging(
        self,
        verbose=False,
    ) -> None:
        # Ensure run_info is available before setting up GCP logging
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
        from ._core.log import logger as bm_logger

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
                    "task_id": self.run_id,  # type: ignore[union-attr] # Checked above
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
        for logger_str in list(logging.Logger.manager.loggerDict.keys()):
            try:
                logging.getLogger(logger_str).setLevel(logging.WARNING)
            except:
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
            message, extra=dict(run=self.run_info.model_dump() if self.run_info else {}),
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
            raise RuntimeError(
                "Logger config with GCP project needed for GCS Log Client.",
            )

        if self._gcs_log_client_cached is None:
            self._gcs_log_client_cached = google.cloud.logging.Client(
                project=self.logger_cfg.project,
                credentials=self._gcp_credentials_cached,  # type: ignore[attr-defined]  # Pass credentials
            )
        return self._gcs_log_client_cached

    @property
    def gcs(self) -> storage.Client:
        # Ensure _gcp_project is set (happens in _gcp_credentials getter)
        if not hasattr(self, "_gcp_project"):
            _ = self._gcp_credentials  # Trigger credential loading if not done yet
            if not hasattr(self, "_gcp_project"):
                raise RuntimeError(
                    "GCP project not determined. Ensure GCP cloud config is present.",
                )

        if self._gcs_cached is None:
            self._gcs_cached = storage.Client(
                project=self._gcp_project, credentials=self._gcp_credentials_cached,
            )  # Pass credentials
        return self._gcs_cached

    @property
    def secret_manager(self) -> SecretsManager:
        if self._secret_manager_cached is None:
            if not self.secret_provider:
                raise RuntimeError("Secret provider configuration is missing.")

            self._secret_manager_cached = SecretsManager(
                **self.secret_provider.model_dump(),
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
                connections = self.secret_manager.get_secret(
                    cfg_key=_MODELS_CFG_KEY,
                )
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
        if not hasattr(self, "_gcp_project"):
            _ = self._gcp_credentials  # Trigger credential loading if not done yet
            if not hasattr(self, "_gcp_project"):
                raise RuntimeError(
                    "GCP project not determined. Ensure GCP cloud config is present.",
                )

        if self._bq_cached is None:
            self._bq_cached = bigquery.Client(
                project=self._gcp_project, credentials=self._gcp_credentials_cached,
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
        job = self.bq.query(sql, job_config=job_config)

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
        result = job.result()
        if return_df:
            if result.total_rows and result.total_rows > 0:
                results_df = result.to_dataframe()
                return results_df
            return pd.DataFrame()
        return result  # Return Any (BigQuery result object)


if __name__ == "__main__":
    pass
