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

from .log import logger

#########
# Agent
#
# A simple class with a function that process a job.
#
# It takes a Job with Inputs and returns a Job with Inputs and Outputs.
# The completed Job is stored in a database (BigQuery) for tracing and analysis.
#
# The primary type of Job is a "flow" which is a sequence of steps that process data
# using a model or client of some sort. In the standard implementation, this is a
# langchain based template processed by an interchangeable LLM Chat model.
#
##########


def get_agent_name_tracing(call: Any) -> str:
    try:
        name = f"{call.inputs['self'].name}: {call.inputs['job'].record.metadata.get('name', call.inputs['job'].record.record_id)}"
        return name
    except:
        return "unknown flow"


class AgentConfig(BaseModel):
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


class AgentVariants(AgentConfig):
    variants: dict[str, Any] = Field(
        default={},
        description="A set of initialisation parameters that will be multiplied together to create individual variant agents.",
    )
    num_runs: int = 1

    model_config = {"extra": "allow"}

    def get_variant_configs(self) -> list[AgentConfig]:
        static_vars = self.model_dump(exclude={"variants", "num_runs"})

        # Create variants (permutations of vars multiplied by num_runs)

        variant_configs = self.num_runs * expand_dict(self.variants)
        variants = []
        for cfg in variant_configs:
            variant = AgentConfig(**static_vars)
            variant.parameters.update(cfg)
            variants.append(variant)

        return variants


class Agent(AgentConfig):
    """Receive data, processes it, save the results, yield, and acknowledge completion."""

    save: SaveInfo | None = Field(default=None)  # Where to save the results
    concurrency: int = Field(default=3)  # Max number of async tasks to run
    data: list[DataSource] | None = []

    _semaphore: asyncio.Semaphore = PrivateAttr(default=None)

    @model_validator(mode="after")
    def setup_semaphore(self) -> "Agent":
        self._semaphore = asyncio.Semaphore(self.concurrency)
        return self

    _semaphore: Semaphore = PrivateAttr(default=None)

    @model_validator(mode="after")
    def setup_semaphore(self) -> "Agent":
        self._semaphore = Semaphore(self.concurrency)
        return self

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

    @model_validator(mode="after")
    def add_extra_params(self) -> "Agent":
        if self.model_extra:
            self.parameters.update(self.model_extra)

        return self

    @field_validator("save", mode="before")
    def validate_save_params(cls, value: SaveInfo | Mapping | None) -> SaveInfo | None:
        if value is None or isinstance(value, SaveInfo):
            return value
        return SaveInfo(**value)

    @trace
    @weave.op(call_display_name=get_agent_name_tracing)
    @workflow(name="run_agent")
    async def run(self, job: Job, **kwargs) -> Job:
        try:
            job.agent_info = self.model_dump(mode="json")
            async with self._semaphore:
                job = await self.process_job(job=job, **kwargs)
        except Exception as e:
            job.error = extract_error_info(e=e)
            if job.record:
                logger.error(
                    f"Error processing task {self.name} with job {job.job_id} and record {job.record.record_id}. Error: {e or type(e)} {e.args=}",
                )
        finally:
            if self.save:
                save_job(job=job, save_info=self.save)
        return job

    async def process_job(
        self,
        *,
        job: Job,
        **kwargs,
    ) -> Job:
        """Take a Job with Inputs, process it, and return a Job with result in Outputs field OR a Job with non-null Error field.

        Inputs:
            job: Job with Inputs
            **kwargs: Additional variables to pass to the agent

        Outputs:
            Job with Inputs and Outputs OR Job with non-null Error field.

        """
        raise NotImplementedError
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
