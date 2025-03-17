import asyncio
import datetime
from asyncio import Semaphore
from collections.abc import Mapping
from typing import Any

import numpy as np
import weave
from omegaconf import DictConfig, ListConfig, OmegaConf
from promptflow.tracing import trace
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from traceloop.sdk.decorators import workflow

from buttermilk._core.agent import AgentInput, AgentOutput
from buttermilk._core.config import DataSource, SaveInfo
from buttermilk.utils.errors import extract_error_info
from buttermilk.utils.save import save
from buttermilk.utils.utils import expand_dict
from buttermilk.utils.validators import convert_omegaconf_objects

import base64
import datetime
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pydantic
import shortuuid
from cloudpathlib import CloudPath
from langchain_core.messages import BaseMessage, HumanMessage
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)

from buttermilk import logger
from buttermilk.llms import LLMCapabilities
from buttermilk.utils.validators import convert_omegaconf_objects, make_list_validator

from .types import SessionInfo

from .log import logger


from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel

##################################
# A single unit of work, including
# a result once we get it.
#
# A Job contains all the information that we need to log
# to know about an agent's operation on a single datum.
#
##################################
class Job(BaseModel):
    job_id: str = pydantic.Field(
        default_factory=shortuuid.uuid,
        description="A unique identifier for this particular unit of work",
    )
    flow_id: str
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC),
        description="The date and time a job was created.",
    )
    source: Sequence[str] = Field(
        ...,
        description="Where this particular job came from",
    )

    run_info: SessionInfo | None = Field(
        default=None,
        description="Information about the context in which this job runs",
    )

    inputs: AgentInput | None = Field(
        default=None,
        description="The data the job will process.",
    )

    # These fields will be fully filled once the record is processed
    agent_info: dict | None = Field(default=None)
    outputs: AgentOutput | None = Field(default=None)
    error: dict[str, Any] = Field(default={})
    metadata: dict | None = Field(default={})

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={
            np.bool_: bool,
            datetime.datetime: lambda v: v.isoformat(),
            ListConfig: lambda v: OmegaConf.to_container(v, resolve=True),
            DictConfig: lambda v: OmegaConf.to_container(v, resolve=True),
        },
        validate_assignment=True,
        exclude_unset=True,
        exclude_none=True,
    ) # type: ignore

    _ensure_list = field_validator("source", mode="before")(
        make_list_validator(),
    )
    _convert = field_validator("outputs", "inputs", "parameters", mode="before")(
        convert_omegaconf_objects(),
    )

    @model_validator(mode="after")
    def move_metadata(self) -> Self:
        if self.outputs and hasattr(self.outputs, "metadata") and self.outputs.metadata:
            if self.metadata is None:
                self.metadata = {}
            self.metadata["outputs"] = self.outputs.metadata
            self.outputs.metadata = None

        return self
