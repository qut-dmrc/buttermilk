import copy
import datetime
from collections.abc import Mapping
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from promptflow.tracing import trace
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

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


class Agent(BaseModel):
    """Receive data, processes it, save the results, yield, and acknowledge completion."""

    name: str = Field(
        ...,
        description="The name of the flow or step in the process that this agent does.",
    )
    save: SaveInfo | None = Field(default=None)  # Where to save the results
    num_runs: int = 1
    concurrency: int = Field(default=4)  # Max number of async tasks to run

    inputs: dict[str, str | list | dict] | None = Field(
        default_factory=dict,
    )

    outputs: dict[str, str | list | dict] | None = Field(
        default_factory=dict,
        description="Data to pass on to next steps.",
    )
    parameters: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Combinations of variables to pass to process job",
        validation_alias=AliasChoices(
            "parameters",
            "params",
            "variants",
            "vars",
            "init",
        ),
    )
    inputs: dict[str, str | list | dict] | None = Field(
        default_factory=dict,
    )

    data: list[DataSource] | None = Field(default_factory=list)

    outputs: dict[str, str | list | dict] | None = Field(
        default_factory=dict,
        description="Data to pass on to next steps.",
    )

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

    # _ensure_list = field_validator("data", mode="before")(make_list_validator())

    _convert = field_validator("outputs", "inputs", "parameters", mode="before")(
        convert_omegaconf_objects(),
    )

    @model_validator(mode="after")
    def add_extra_params(self) -> "Agent":
        if self.model_extra:
            self.parameters.update(self.model_extra)
        return self

    def make_combinations(self, **extra_combinations: dict):
        # Because we're duplicating variables and returning
        # permutations, we should make sure to return a copy,
        # not the original.
        params = self.parameters.copy()
        params.update(extra_combinations)
        vars = self.num_runs * expand_dict(params)
        return copy.deepcopy(vars)

    @field_validator("save", mode="before")
    def validate_save_params(cls, value: SaveInfo | Mapping | None) -> SaveInfo | None:
        if value is None or isinstance(value, SaveInfo):
            return value
        return SaveInfo(**value)

    @trace
    async def run(self, job: Job, **kwargs) -> Job:
        try:
            job.agent_info = self.model_dump(mode="json")
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

    def make_combinations(self):
        # Because we're duplicating variables and returning
        # permutations, we should make sure to return a copy,
        # not the original.
        vars = self.num_runs * expand_dict(self.parameters)
        return copy.deepcopy(vars)

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
