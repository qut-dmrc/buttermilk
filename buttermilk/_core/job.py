import datetime
from collections.abc import Sequence
from typing import Any, Self

import pydantic
import shortuuid
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from buttermilk.utils.validators import convert_omegaconf_objects, make_list_validator

from .types import SessionInfo


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

    inputs: "AgentInput | None" = Field(
        default=None,
        description="The data the job will process.",
    )

    # These fields will be fully filled once the record is processed
    agent_info: dict | None = Field(default=None)
    outputs: "AgentOutput | None" = Field(default=None)
    error: dict[str, Any] = Field(default={})
    metadata: dict | None = Field(default={})

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        use_enum_values=True,
        # json_encoders={
        #     np.bool_: bool,
        #     datetime.datetime: lambda v: v.isoformat(),
        #     ListConfig: lambda v: OmegaConf.to_container(v, resolve=True),
        #     DictConfig: lambda v: OmegaConf.to_container(v, resolve=True),
        # },
        validate_assignment=True,
        exclude_unset=True,
        exclude_none=True,
    )  # type: ignore

    _ensure_list = field_validator("source", mode="before")(
        make_list_validator(),
    )
    _convert = field_validator("outputs", "inputs", mode="before")(
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
