from typing import Any, AsyncGenerator, Generator, Literal, Optional, Self, Sequence, Tuple, Type, TypeVar, Union,Mapping
from cloudpathlib import CloudPath, GSPath
from pydantic import (
    AliasChoices,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    field_validator,
    model_validator,
)

from hydra_zen import instantiate, builds

from .agent import Agent


class DataSource(BaseModel):
    name: str
    max_records_per_group: int = -1
    type: Literal["job", "file", "bq", "generator"]
    path: str = Field(..., validation_alias=AliasChoices("path", "dataset", "uri", "func"))
    filter: Optional[Mapping[str, str]] = Field(default_factory=dict)
    join: Optional[Mapping[str, str]] = Field(default_factory=dict)
    agg: Optional[bool] = Field(default=False)
    group: Optional[Mapping[str, str]] = Field(default_factory=dict)
    columns: Optional[Mapping[str, str]] = Field(default_factory=dict)

class Flow(BaseModel):
    name: str
    num_runs: int = 1
    concurrency: int = 1
    agent: Agent
    data: Optional[Sequence[DataSource]] = Field(default_factory=list)
    parameters: Optional[Mapping] = Field(default_factory=dict)

class Tracing(BaseModel):
    enabled: bool = False
    endpoint: Optional[str] = None
    otlp_headers: Optional[Mapping] = Field(default_factory=dict)


CloudProvider = TypeVar(Literal["gcp", "aws", "azure", "hashicorp", "env", "vault", "local"])
class CloudProviderCfg(BaseModel):
    type: CloudProvider
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, populate_by_name=True, exclude_none=True, exclude_unset=True, include_extra=True)

class RunCfg(BaseModel):
    platform: str = 'local'
    parameters: Mapping[str, Any] = Field(default_factory=dict)

class Project(BaseModel):
    name: str
    job: str
    connections: Sequence[str] = Field(default_factory=list)
    secret_provider: CloudProviderCfg
    save_dest: CloudProviderCfg
    logger: CloudProvider
    flows: list[Flow] = Field(default_factory=list)
    tracing: Optional[Tracing] = Field(default_factory=Tracing)
    verbose: bool = True
    cloud: list[CloudProviderCfg] = Field(default_factory=list)
    run: RunCfg

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, populate_by_name=True,          exclude_none=True, exclude_unset=True,)



# @hydra.main(version_base="1.3", config_path="../conf", config_name="config")
# def main(cfg: DictConfig) -> None:
#     validated_config = instantiate(Config, cfg)
#     print(validated_config)

if __name__ == '__main__':
    Config = builds(Project, populate_full_signature=True)
    Config
    pass