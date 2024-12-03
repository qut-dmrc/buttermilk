import copy
import datetime
from collections.abc import Mapping
from typing import Any, Self, Sequence

import numpy as np
import pandas as pd
import shortuuid
from omegaconf import DictConfig, ListConfig, OmegaConf
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
from buttermilk.utils.utils import expand_dict, find_in_nested_dict

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
    save: SaveInfo | None = Field(None)  # Where to save the results
    parameters: dict[str, str | list | dict] | None = Field(
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
    datasets: dict = Field(
        default_factory=dict,
    )
    outputs: dict[str, str | list | dict] | None = Field(
        default_factory=dict,
        description="Data to pass on to next steps.",
    )
    _agent_id: str | None = PrivateAttr(None)

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
        self._agent_id = f"{self.name}_{shortuuid.uuid()[:6]}"
        return self

    @trace
    async def run(self, *, job: Job, **kwargs) -> Job:
        try:
            job.agent_info = self.model_dump(mode="json")
            job = await self.process_job(job=job, **kwargs)
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
                elif self.save:
                    save(data=rows, save_dir=self.save.destination)

        return job
    
    async def prepare_inputs(self,
        job: Job,
        additional_data: dict = None,
        **kwargs,
    ): 
        
        # Process all inputs into two categories.
        # Job objects have a .params mapping, which is usually the result of a combination of init variables that will be common to multiple runs over different records.
        # Job objects also have a .inputs mapping, which is the result of a combination of inputs that will be unique to a single record.
        # Then there are also extra **kwargs sent to this method.
        # In all cases, input values might be the name of a template, a literal value, or a reference to a field in the job.record object or in other supplied additional_data.
        # We need to resolve all inputs into a mapping that can be passed to the agent.

        # First, log that we received extra **kwargs
        job.inputs.update(**kwargs)

        # Create a dictionary for complete prompt messages that we will not pass to the templating function
        placeholders = {}
        placeholders["record"] = job.record

        # And combine all sources of inputs into one dict
        all_params = {**job.parameters, **job.inputs}

        # but remove 'template', we deal with that explicitly, it's always required.
        _ = all_params.pop("template", None)

        input_vars = {}
        for key, value in all_params.items():
            if not (
                resolved_value := resolve_value(
                    value, job, additional_data=additional_data
                )
            ):
                continue
            if value == "record":  # Special case for full record placeholder
                placeholders[key] = resolved_value
            else:
                input_vars[key] = resolved_value
        return input_vars, placeholders
    
    async def process_job(
        self,
        *,
        job: Job,
        additional_data: Any = None,
        dataset: pd.DataFrame = None,
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

        # Todo: move this to the class properly
        input_vars, placeholders = await self.prepare_inputs(
            job=job,
            additional_data=additional_data,
            **kwargs,
        )
        raise NotImplementedError


def resolve_value(value, job, additional_data):
    """Recursively resolve values from different data sources."""
    if isinstance(value, str):
        # Handle special "record" case
        if value.lower() == "record":
            return job.record

        # Handle dot notation
        if "." in value:
            locator, field = value.split(".", maxsplit=1)
            if additional_data and locator in additional_data:
                if isinstance(additional_data[locator], pd.DataFrame):
                    return additional_data[locator][field].values
                return find_in_nested_dict(additional_data[locator], field)
            if locator == "record":
                return find_in_nested_dict(job.record.model_dump(), field)

        # Handle direct record field reference
        if job.record and (
            value in job.record.model_fields or value in job.record.model_extra
        ):
            return getattr(job.record, value)

        # handle entire dataset
        if additional_data and value in additional_data:
            if isinstance(additional_data[value], pd.DataFrame):
                return additional_data[value].astype(str).to_dict(orient="records")
            return additional_data[value]

        # No match
        return value

    if isinstance(value, Sequence) and not isinstance(value, str):
        # combine lists
        return [
            x
            for item in value
            for x in resolve_value(item, job, additional_data=additional_data)
        ]

    if isinstance(value, dict):
        return {
            k: resolve_value(v, job, additional_data=additional_data)
            for k, v in value.items()
        }

    return value
