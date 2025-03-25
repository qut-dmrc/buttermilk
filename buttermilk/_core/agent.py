from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Self

import pydantic
import weave
from pydantic import (
    BaseModel,
    Field,
)

from buttermilk._core.config import DataSource, SaveInfo
from buttermilk._core.contract import AgentInput, AgentMessages, AgentOutput, UserInput
from buttermilk.utils.save import save


class ToolConfig(BaseModel):
    id: str
    name: str
    description: str
    tool_obj: str | None = None

    data_cfg: list[DataSource] = Field(
        default=[],
        description="Specifications for data that the Agent should load",
    )

    def _run(self, *args, **kwargs):
        raise NotImplementedError


#########
# Agent
#
# A simple class with a function that process a job.
#
# It takes a Job with Inputs and returns a Job with Inputs and Outputs.
# The completed Job is stored in a database (BigQuery) for tracing and analysis.
#
##########


class AgentConfig(BaseModel):
    agent_obj: str = Field(
        ...,
        description="The object name to instantiate",
    )
    id: str = Field(
        ...,
        description="The unique name of this agent.",
    )
    name: str = Field(
        ...,
        description="A human-readable name for this agent.",
    )
    description: str = Field(
        ...,
        description="Short explanation of what this agent type does",
    )
    tools: list[ToolConfig] = Field(
        default=[],
        description="Tools the agent can invoke",
    )
    data: list[DataSource] = Field(
        default=[],
        description="Specifications for data that the Agent should load",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Initialisation parameters to pass to the agent",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="A mapping of data to agent inputs",
    )
    outputs: dict[str, Any] = {}
    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": False,
        "populate_by_name": True,
    }


class Agent(AgentConfig, ABC):
    """Base Agent interface for all processing units"""

    agent_obj: str = Field(
        default="",
        description="The object name to instantiate",
        exclude=True,
    )

    _trace_this = True
    _run_fn: Callable | Awaitable

    @pydantic.model_validator(mode="after")
    def _get_process_func(self) -> Self:
        """Returns the appropriate processing function based on tracing setting."""

        def _process_fn():
            if self._trace_this:
                return weave.op(self._process, call_display_name=f"{self.id}")
            return self._process

        self._run_fn = _process_fn()
        return self

    # @workflow(name="run_agent")
    # @trace

    @abstractmethod
    async def receive_output(
        self,
        message: AgentMessages | UserInput,
        source: str,
        **kwargs,
    ) -> AgentMessages | None:
        """Log data or send output to the user interface"""
        raise NotImplementedError

    @abstractmethod
    async def _process(self, input_data: AgentInput, **kwargs) -> AgentOutput | None:
        """Process input data and return output

        Inputs:
            input_data: AgentInput with appropriate values required by the agent.

        Outputs:
            AgentOutput record with processed data or non-null Error field.

        """
        raise NotImplementedError
        return job

    async def __call__(self, input_data: AgentInput, **kwargs) -> AgentOutput | None:
        """Allow agents to be called directly as functions"""
        return await self._run_fn(input_data, **kwargs)

    async def initialize(self, **kwargs) -> None:
        """Initialize the agent"""


def save_job(job: "Job", save_info: SaveInfo) -> str:
    rows = [job.model_dump(mode="json", exclude_none=True)]
    if save_info.type == "bq":
        dest = save(
            data=rows,
            dataset=save_info.dataset,
            schema=save_info.db_schema,
            save_dir=save_info.destination,
        )
    else:
        dest = save(data=rows, save_dir=save_info.destination)

    return dest
