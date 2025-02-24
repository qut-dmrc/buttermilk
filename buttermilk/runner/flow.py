import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Self

import copy
import pandas as pd
import weave
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from traceloop.sdk.decorators import workflow

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.config import DataSource
from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.exceptions import FatalError
from buttermilk.runner.helpers import (
    combine_datasets,
    parse_flow_vars,
    prepare_step_df,
)
from buttermilk.utils.utils import expand_dict
from buttermilk.utils.validators import make_list_validator

""" A flow ties several stages together, runs them over a single record, 
    and streams results.
"""


def get_flow_name_tracing(call: Any) -> str:
    try:
        name = f"{call.inputs['self'].source}: {call.inputs['job'].record.metadata.get('name', call.inputs['job'].record.record_id)}"
        return name
    except:
        return "unknown flow"


class Flow(BaseModel):
    source: Sequence[str]
    steps: list[Agent]

    data: list[DataSource] | None = Field(default_factory=list)
    _data: dict = {}
    _results: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)

    class Config:
        arbitrary_types_allowed = True

    _ensure_list = field_validator("source", mode="before")(
        make_list_validator(),
    )

    # @field_validator("data", mode="before")
    # def convert_data(cls, value):
    #     datasources = []
    #     for source in value:
    #         if not isinstance(source, DataSource):
    #             source = DataSource(**source)
    #         datasources.append(source)
    #     return datasources

    @model_validator(mode="after")
    def combine_datasets(self) -> Self:
        self._results = combine_datasets(
            existing_df=self._results,
            datasources=self.data,
        )
        for agent in self.steps:
            # Create empty list for results from each step
            self._data[agent.name] = self._data.get(agent.name, [])

        # # this is a hack, it wont work later:
        # self._data[self.data[0].name] = self._results.to_dict(
        #     orient="records",
        #     index=True,
        # )
        return self

    def get_record(self, record_id: str) -> RecordInfo:
        rec = self._results.query("record_id==@record_id")
        if rec.shape[0] > 1:
            raise ValueError(
                f"More than one record found for query record_id == {record_id}",
            )

        return RecordInfo(**rec.iloc[0].to_dict())

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
        job.source = list(set(self.source + job.source)) if job.source else self.source

        for agent in self.steps:
            async for result in self.run_step_variants(
                agent=agent,
                job=job,
            ):
                if result.record:
                    job.record = result.record
                yield result

        return

    # @workflow(name="run_step")
    # @weave.op(call_display_name=get_flow_name_tracing)
    async def run_step_variants(
        self,
        *,
        agent: Agent,
        job: Job,
    ) -> AsyncGenerator:
        # Store base components of the job that don't change for each variant in the permutations
        jobvars = job.model_dump(
            exclude={
                "job_id": True,
                "parameters": True,
                "inputs": True,
                "outputs": True,
                "error": True,
                "metadata": True,
                "timestamp": True,
                "identifier": True,
                "record": {"fulltext": True},
            },
        )


        # Because we're duplicating variables and returning
        # permutations, we should make sure to edit a copy,
        # not the original.
        params = copy.deepcopy(agent.parameters)
        inputs = copy.deepcopy(agent.inputs)

        # In all cases, input values might be the name of a template, a literal value, 
        # or a reference to a field in the job.record object or in other supplied additional_data.
        # We need to resolve all flow inputs into a mapping that can be passed to the agent.
        # Makesure we make all job variables accessible for mapping but don't pass the job dict by
        # reference (this includes the data record in job.record and the computed fields like 'fulltext')
        params = parse_flow_vars(
            params,
            flow_data=job.model_dump(),
            additional_data=self._data,
        )
        inputs = parse_flow_vars(
            inputs,
            flow_data=job.model_dump(),
            additional_data=self._data,
        )

        # Create a new job and task for every combination of variables
        # this agent is configured to run. Agents have a parameters mapping; 
        # each permutation of these is multiplied by num_runs. Agents also
        # have an inputs mapping that does not get multiplied.
        variants = agent.num_runs * expand_dict(params)
        
        tasks = []
        for i in range(len(variants)):
            variants[i].update(inputs)
            job_variant = Job(**jobvars, parameters=variants[i])

            task = agent.run(
                job=job_variant,
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
                        if agent.outputs:
                            # Process result as specified in the output map field of Flow
                            result.outputs = parse_flow_vars(
                                agent.outputs,
                                flow_data=result.model_dump(),
                                additional_data=self._data,
                            )
                        # incorporate successful runs into data store for future use
                        self.incorporate_outputs(
                            step_name=agent.name,
                            outputs=result.outputs,
                        )
                    except Exception as e:
                        # log the error but do not abort.
                        error_msg = f"Agent {agent.name} response data not formatted as expected: {e}"
                        logger.error(error_msg)
                        result.error = dict(
                            message=error_msg,
                            type=type(e).__name__,
                            args=e.args,
                        )

                    # We are in the process of replacing this with a single dataframe
                    # that holds the progressive results of the entire flow.
                    # results_df = pd.DataFrame.from_records(outputs)
                    # self._results = combine_datasets(
                    #     existing_df=self._results,
                    #     datasources=agent.data,
                    #     results_df=results_df,
                    # )

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
        outputs: dict,
    ) -> None:
        """Update the data object with the outputs of the agent."""
        if isinstance(outputs, BaseModel):
            dict_outputs = outputs.model_dump()
        else:
            dict_outputs = outputs
        if step_name not in self._data:
            self._data[step_name] = []

        self._data[step_name].append(dict_outputs)
