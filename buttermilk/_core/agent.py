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

from buttermilk._core.config import DataSource, SaveInfo
from buttermilk._core.runner_types import Job
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

class AgentInput(BaseModel):
    """Base class for agent inputs with built-in validation"""
    pass


class AgentOutput(BaseModel):
    """Base class for agent outputs with built-in validation"""
    pass

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
    
    data: list[DataSource] | None = []
    agent: str = Field(..., description="The object to instantiate")
    name: str = Field(
        ...,
        description="The name of the flow or step in the process that this agent does.",
    )
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Initialisation parameters to pass to the agent",
    )
    data: list[DataSource] | None = Field(default_factory=list)
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="A mapping of data to agent inputs",
    )
    outputs: dict[str, Any] = {}
    
    _convert_params = field_validator("outputs", "inputs", "parameters", mode="before")(
        convert_omegaconf_objects(),
    )
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = False
        populate_by_name = True
        exclude_none = True
        exclude_unset = True

        json_encoders = {
            np.bool_: bool,
            datetime.datetime: lambda v: v.isoformat(),
            ListConfig: lambda v: OmegaConf.to_container(v, resolve=True),
            DictConfig: lambda v: OmegaConf.to_container(v, resolve=True),
        }

    @field_validator("save", mode="before")
    def validate_save_params(cls, value: SaveInfo | Mapping | None) -> SaveInfo | None:
        if value is None or isinstance(value, SaveInfo):
            return value
        return SaveInfo(**value)
    
    @abstractmethod
    async def process(self, input_data: AgentInput) -> AgentOutput:
        """Process input data and return output

        Inputs:
            job: Job with Inputs
            **kwargs: Additional variables to pass to the agent

        Outputs:
            Job with Inputs and Outputs OR Job with non-null Error field.

        """
        raise NotImplementedError
        return job
        
    async def __call__(self, input_data: AgentInput) -> AgentOutput:
        """Allow agents to be called directly as functions"""
        return await self.process(input_data)
    
    @classmethod
    def get_variant_configs(cls, *, variants: dict[str, Any], num_runs: int = 1, **kwargs) -> list[Agent]:
        """A factory for creating Agent instance variants for a single
        step of a workflow.

        Creates a new agent for every combination of parameters in a given
        step of the workflow to run. Agents have a variants mapping;
        each permutation of these is multiplied by num_runs. Agents also
        have an inputs mapping that does not get multiplied.
        """
        
        # Create variants (permutations of vars multiplied by num_runs)

        variant_configs = num_runs * expand_dict(variants)
        agents = []
        for cfg in variant_configs:
            variant = dict(**kwargs)
            variant['parameters'].update(cfg)
            agents.append(cls(**variant))

        return agents

    @trace
    @weave.op(call_display_name=get_agent_name_tracing)
    @workflow(name="run_agent")
    async def run(self, job: Job, **kwargs) -> Job:
        try:
            job.agent_info = self.model_dump(mode="json")
            job = await self.process(input_data=job, **kwargs)
        except Exception as e:
            job.error = extract_error_info(e=e)
            if job.record:
                logger.error(
                    f"Error processing task {self.name} with job {job.job_id} and record {job.record.record_id}. Error: {e or type(e)} {e.args=}",
                )
        return job



def save_job(job: Job, save_info: SaveInfo) -> str:
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



