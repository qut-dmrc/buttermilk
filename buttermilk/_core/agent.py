import functools
from abc import ABC, abstractmethod
from typing import Any

import weave
from pydantic import (
    BaseModel,
    Field,
)

from buttermilk._core.config import SaveInfo
from buttermilk._core.contract import AgentInput, AgentMessages, AgentOutput
from buttermilk.utils.save import save


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

    def _get_process_func(self):
        """Returns the appropriate processing function based on tracing setting."""
        if self._trace_this:
            return weave.op(call_display_name=f"{self.id}")(
                functools.partial(self._process),
            )
        return self._process

    # @workflow(name="run_agent")
    # @trace

    @abstractmethod
    async def receive_output(
        self,
        message: AgentMessages,
        source: str,
        **kwargs,
    ) -> AgentMessages | None:
        """Log data or send output to the user interface"""
        raise NotImplementedError

    @abstractmethod
    async def _process(self, input_data: AgentMessages, **kwargs) -> AgentOutput | None:
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
        process_func = self._get_process_func()
        return await process_func(input_data, **kwargs)

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
