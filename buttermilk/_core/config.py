import datetime
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    Literal,
    Self,
)

import cloudpathlib
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from buttermilk.defaults import BQ_SCHEMA_DIR

BASE_DIR = Path(__file__).absolute().parent


# from functools import wraps
# from typing import Any, Callable
# from pydantic import BaseModel

# async def run_flow(job: Job, flow: callable):
#     response = await flow.call_async(**job.inputs)
#     job.outputs = Result(**response)
#     return job

# def flow():
#     def inner(func):
#         @wraps(func)
#         def _impl(job: Job) -> Job:
#             return run_flow(job, func)
#         return _impl


CloudProvider = Literal["gcp", "bq", "aws", "azure", "env", "local", "gsheets"]


class CloudProviderCfg(BaseModel):
    type: CloudProvider

    class Config:
        # Exclude fields with None values when serializing
        exclude_none = True
        arbitrary_types_allowed = True
        populate_by_name = True
        # Ignore extra fields not defined in the model
        extra = "allow"
        exclude_none = True
        exclude_unset = True
        include_extra = True


class SaveInfo(CloudProviderCfg):
    destination: str | cloudpathlib.AnyPath | None = None
    db_schema: str | None = Field(
        default=None,
        validation_alias=AliasChoices("db_schema", "schema"),
    )
    dataset: str | None = Field(default=None)

    model_config = ConfigDict(
        json_encoders={
            np.bool_: bool,
            datetime.datetime: lambda v: v.isoformat(),
            ListConfig: lambda v: OmegaConf.to_container(v, resolve=True),
            DictConfig: lambda v: OmegaConf.to_container(v, resolve=True),
        },
    )

    @field_validator("db_schema")
    def file_must_exist(cls, v):
        if v:
            if not os.path.exists(v):
                f = Path(BQ_SCHEMA_DIR) / v
                if f.exists():
                    return f.as_posix()

                raise ValueError(f"File '{v}' does not exist.")
        return v

    @model_validator(mode="after")
    def check_destination(self) -> Self:
        if not self.destination and not self.dataset:
            raise ValueError(
                "Nowhere to save to! Either destination or dataset must be provided.",
            )
        return self


class DataSource(BaseModel):
    name: str
    max_records_per_group: int = -1
    type: Literal["job", "file", "bq", "generator", "plaintext"]
    path: str = Field(
        ...,
        validation_alias=AliasChoices("path", "dataset", "uri", "func"),
    )
    glob: str = Field(default="**/*")
    filter: Mapping[str, str | Sequence[str] | None] | None = Field(
        default_factory=dict,
    )
    join: Mapping[str, str] | None = Field(default_factory=dict)
    agg: bool | None = Field(default=False)
    group: Mapping[str, str] | None = Field(default_factory=dict)
    columns: Mapping[str, str | Mapping] | None = Field(default_factory=dict)
    last_n_days: int = Field(default=7)

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = False
        populate_by_name = True
        exclude_none = True
        exclude_unset = True


class Tracing(BaseModel):
    enabled: bool = False
    endpoint: str | None = None
    otlp_headers: Mapping | None = Field(default_factory=dict)


class RunCfg(BaseModel):
    platform: str = "local"
    max_concurrency: int = -1
    parameters: Mapping[str, Any] = Field(default_factory=dict)
