import asyncio
import copy
from collections.abc import Mapping
from typing import Any, Self

import shortuuid
from promptflow.tracing import trace
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from buttermilk._core.config import SaveInfo
from buttermilk._core.runner_types import Job
from buttermilk.utils.errors import extract_error_info
from buttermilk.utils.save import save
from buttermilk.utils.utils import expand_dict

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

    name: str
    num_runs: int = 1
    concurrency: int = Field(default=4)  # Max number of async tasks to run
    save: SaveInfo | None = Field(default=None)  # Where to save the results
    parameters: dict[str, Any] | None = Field(
        default=dict,
        description="Combinations of variables to  to pass to process job",
        validation_alias=AliasChoices(
            "parameters",
            "params",
            "variants",
            "vars",
            "init",
        ),
    )
    inputs: dict[str, Any] = Field(default=dict)
    outputs: dict[str, Any] | None = Field(
        default=dict,
        description="Data to pass on to next steps.",
    )
    _agent_id: str = PrivateAttr(None)
    _sem: asyncio.Semaphore = PrivateAttr()  # Semaphore for limiting concurrent tasks

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True
        populate_by_name = True
        exclude_none = True
        exclude_unset = True

    @model_validator(mode="after")
    def add_extra_params(self) -> Self:
        if self.model_extra:
            self.parameters.update(self.model_extra)
        return self

    def make_combinations(self):
        # Because we're duplicating variables and returning
        # permutations, we should make sure to return a copy,
        # not the original.
        vars = self.num_runs * expand_dict(self.parameters)
        return copy.deepcopy(vars)

    @field_validator("save", mode="before")
    def validate_save_params(cls, value: SaveInfo | Mapping | None) -> SaveInfo | None:
        if value is None or isinstance(value, SaveInfo):
            return value
        return SaveInfo(**value)

    @model_validator(mode="after")
    def validate_concurrency(self) -> Self:
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._sem = asyncio.Semaphore(value=self.concurrency)
        self._agent_id = f"{self.name}_{shortuuid.uuid()[:6]}"
        return self

    @trace
    async def run(self, *, parameters: dict, job: Job) -> Job:
        async with self._sem:
            try:
                job.agent_info = self.model_dump()
                job = await self.process_job(job=job, vars=parameters)
            except Exception as e:
                job.error = extract_error_info(e=e)
                if job.record:
                    logger.error(
                        f"Error processing task {self.name} by {self.name} with job {job.job_id} and record {job.record.record_id}. Error: {e or type(e)} {e.args=}",
                    )
            finally:
                if self.save:
                    rows = [job.model_dump()]
                    if self.save.type == "bq":
                        save(
                            data=rows,
                            dataset=self.save.dataset,
                            schema=self.save.db_schema,
                            save_dir=self.save.destination,
                        )
                    else:
                        save(data=rows, save_dir=self.save.destination)
            return job

    async def process_job(
        self,
        *,
        job: Job,
        additional_data: Any = None,
        **kwargs,
    ) -> Job:
        """Take a Job with Inputs, process it, and return a Job.

        Inputs:
            job: Job with Inputs
            additional_data: Any additional data to pass to the agent
            **kwargs: Additional variables to pass to the agent

        Outputs:
            Job with Inputs and Outputs

        """
        raise NotImplementedError
