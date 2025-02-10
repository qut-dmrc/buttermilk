import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Self

import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.config import DataSource
from buttermilk._core.runner_types import Job
from buttermilk.exceptions import FatalError
from buttermilk.runner.helpers import (
    combine_datasets,
    parse_flow_vars,
    prepare_step_df,
)
from buttermilk.utils.validators import make_list_validator

""" A flow ties several stages together, runs them over a single record, 
    and streams results.
"""


class Flow(BaseModel):
    source: Sequence[str]
    steps: list[Agent]

    data: list[DataSource] | None = Field(default_factory=list)
    _data: dict = None
    _results: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)

    class Config:
        arbitrary_types_allowed = True

    _ensure_list = field_validator("source", mode="before")(
        make_list_validator(),
    )

    @field_validator("data", mode="before")
    def convert_data(cls, value):
        datasources = []
        for source in value:
            if not isinstance(source, DataSource):
                source = DataSource(**source)
            datasources.append(source)
        return datasources

    @model_validator(mode="after")
    def combine_datasets(self) -> Self:
        self._results = combine_datasets(
            existing_df=self._results,
            datasources=self.data,
        )
        return self

    async def load_data(self):
        # We are in the process of replacing _data with a single
        # _results dataframe.
        self._data = await prepare_step_df(self.data)

    async def run_flows(
        self,
        *,
        job: Job,
    ) -> AsyncGenerator[Any, None]:
        if self._data is None:
            await self.load_data()

        for agent in self.steps:
            async for result in self.run_step(
                agent=agent,
                job=job,
            ):
                if result.record:
                    job.record = result.record
                yield result

        return

    async def run_step(
        self,
        *,
        agent: Agent,
        job: Job,
    ) -> AsyncGenerator:
        self._data[
            agent.name
        ] = {}  # Ensure this line is present to initialize _data for the agent

        job.source = list(set(self.source + job.source)) if job.source else self.source

        # Store base components of the job that don't change for each variant in the permutations
        job_vars = job.model_dump(
            exclude=["job_id", "parameters", "timestamp"],
        )
        tasks = []

        # Expand mapped parameters before producing permutations of jobs
        params = parse_flow_vars(
            agent.parameters,
            job=job,
            additional_data=self._data,
        )

        # Create a new job and task for every combination of variables
        # this agent is configured to run.
        for variant in agent.make_combinations(**params):
            # Process all inputs into two categories.
            # Job objects have a .params mapping, which is usually the result of a combination of init variables that will be common to multiple runs over different records.
            # Job objects also have a .inputs mapping, which is the result of a combination of inputs that will be unique to a single record.
            # Then there are also extra **kwargs sent to this method.
            # In all cases, input values might be the name of a template, a literal value, or a reference to a field in the job.record object or in other supplied additional_data.
            # We need to resolve all inputs into a mapping that can be passed to the agent.

            # After this method, job.parameters will include all variables that will be passed
            # during the initital construction of the job - including template variables and static values
            # job.inputs will include all variables and formatted placeholders etc that will not be passed
            # to the templating function and will be sent direct instead
            job_variant = Job(**job_vars, parameters=variant)

            job_variant.inputs = parse_flow_vars(
                agent.inputs,
                job=job_variant,
                additional_data=self._data,
            )

            task = agent.run(
                job=job_variant,
                additional_data=self._data,
            )
            tasks.append(task)

        logger.info(
            f"Starting {len(tasks)} async tasks for {self.__repr_name__()} step {agent.name}",
        )

        # Process and yield the results as they finish
        for task in asyncio.as_completed(tasks):
            try:
                result: Job = await task

                if result.error:
                    # Log errors and yield result without further processing
                    logger.error(
                        f"Agent {agent.name} failed with error: {result.error}",
                    )
                else:
                    try:
                        # Process result as specified in the output map field of Flow
                        result.outputs = parse_flow_vars(
                            agent.outputs,
                            job=result,
                            additional_data=self._data,
                        )

                        # incorporate successful runs into data store for future use
                        self.incorporate_outputs(
                            step_name=agent.name,
                            outputs=result.outputs,
                        )

                        # We are in the process of replacing this with a single dataframe
                        # that holds the progressive results of the entire flow.
                        # results_df = pd.DataFrame.from_records(outputs)
                        # self._results = combine_datasets(
                        #     existing_df=self._results,
                        #     datasources=agent.data,
                        #     results_df=results_df,
                        # )

                    except Exception as e:
                        # log the error but do not abort.
                        error_msg = f"Agent {agent.name} response data not formatted as expected: {e}"
                        logger.error(error_msg)
                        result.error = dict(
                            message=error_msg,
                            type=type(e).__name__,
                            args=e.args,
                        )

                yield result  # Yield result within the loop

            except FatalError as e:  # Handle FatalError to abort the flow
                message = f"Aborting flow -- critical error running task from agent {agent.name}: {e}"
                logger.error(message)
                # Cancel remaining tasks (important for proper cleanup)
                for t in tasks:
                    if not t.done():
                        t.cancel()
                raise FatalError(message) from e  # Re-raise to stop the outer loop

            except Exception as e:  # Handle other exceptions
                msg = (
                    f"Agent {agent.name} hit unknown error in run_flows: {e}, {e.args=}"
                )
                logger.exception(msg)
                # Continue processing for now
        return

    def incorporate_outputs(
        self,
        step_name: str,
        outputs: dict[str, Any],
    ) -> None:
        """Update the data object with the outputs of the agent."""
        if step_name not in self._data:
            self._data[step_name]["outputs"] = [outputs]
        else:
            for k, v in outputs.items():
                # create if key does not already exist
                self._data[step_name][k] = self._data[step_name].get(k, [])
                self._data[step_name][k].append(v)
