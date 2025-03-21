from abc import ABC, abstractmethod
from typing import Any

import weave
from promptflow.tracing import trace
from pydantic import (
    BaseModel,
    Field,
)
from traceloop.sdk.decorators import workflow

from buttermilk import logger
from buttermilk._core.config import SaveInfo
from buttermilk._core.contract import AgentInput, AgentMessages, AgentOutput
from buttermilk.utils.errors import extract_error_info
from buttermilk.utils.save import save

from .log import logger


def get_agent_name_tracing(call: Any) -> str:
    try:
        name = f"{call.inputs['self'].name}: {call.inputs['job'].record.metadata.get('name', call.inputs['job'].record.record_id)}"
        return name
    except:
        return "unknown flow"


#########
# Agent
#
# A simple class with a function that process a job.
#
# It takes a Job with Inputs and returns a Job with Inputs and Outputs.
# The completed Job is stored in a database (BigQuery) for tracing and analysis.
#
##########
class Agent(BaseModel, ABC):
    """Base Agent interface for all processing units"""

    agent_id: str = Field(
        ...,
        description="The unique name of this agent.",
    )
    name: str = Field(..., description="The name of the step this agent type performs")
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Initialisation parameters to pass to the agent",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="A mapping of data to agent inputs",
    )
    outputs: dict[str, Any] = {}

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = False

    @trace
    @weave.op(call_display_name=get_agent_name_tracing)
    @workflow(name="run_agent")
    async def run(self, job: "Job", **kwargs) -> "Job":
        try:
            job.agent_info = self.model_dump(mode="json")
            job = await self.process(input_data=job, **kwargs)
        except Exception as e:
            job.error = extract_error_info(e=e)
            if job.record:
                logger.error(
                    f"Error processing task {self.agent_id} with job {job.job_id} and record {job.record.record_id}. Error: {e or type(e)} {e.args=}",
                )
        return job

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
    async def process(self, input_data: AgentMessages, **kwargs) -> AgentMessages:
        """Process input data and return output

        Inputs:
            input_data: AgentInput with appropriate values required by the agent.

        Outputs:
            AgentOutput record with processed data or non-null Error field.

        """
        raise NotImplementedError
        return job

    async def __call__(self, input_data: AgentInput, **kwargs) -> AgentOutput:
        """Allow agents to be called directly as functions"""
        return await self.process(input_data, **kwargs)


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
