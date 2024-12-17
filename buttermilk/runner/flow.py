import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import (
    Any,
)

from pydantic import BaseModel, Field, field_validator

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.config import DataSource
from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk._core.types import SessionInfo
from buttermilk.runner.helpers import parse_flow_vars, prepare_step_df
from buttermilk.utils.utils import find_in_nested_dict

""" A flow ties several stages together, runs them over a single record, 
    and streams results.
"""


class Flow(BaseModel):
    source: str | Sequence[str] | None
    steps: list[Agent]

    data: list[DataSource] | None = Field(default_factory=list)
    _data: dict = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator("data", mode="before")
    def convert_data(cls, value):
        datasources = []
        for source in value:
            if not isinstance(source, DataSource):
                source = DataSource(**source)
            datasources.append(source)
        return datasources

    async def load_data(self):
        self._data = await prepare_step_df(self.data)

    async def run_flows(
        self,
        *,
        flow_id: str,
        record: RecordInfo | None,
        run_info: SessionInfo,
        source: str | Sequence[str] | None,
        q: str | None = None,
    ) -> AsyncGenerator[Any, None]:

        if self._data is None:
            await self.load_data()

        for agent in self.steps:
            async for result in self.run_step(
                agent=agent,
                record=record,
                run_info=run_info,
                source=source,
                flow_id=flow_id,
                q=q,
            ):
                yield result

    async def run_step(
        self,
        *,
        agent: Agent,
        record: RecordInfo | None,
        flow_id: str,
        run_info: SessionInfo,
        q: str | None = None,
        source: Sequence[str] = [],
    ) -> AsyncGenerator:
        self._data[agent.name] = {}
        source = [self.source, *source] if source else [self.source]
        tasks = []
        for variant in agent.make_combinations():
            # Create a new job and task for every combination of variables
            # this agent is configured to run.
            job = Job(
                flow_id=flow_id,
                record=record,
                source=source,
                run_info=run_info,
            )

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
            job.inputs = parse_flow_vars(agent.inputs, job=job, additional_data=self._data)
            job.parameters = parse_flow_vars(variant, job=job, additional_data=self._data)
            task = agent.run(
                job=job,
                q=q,
                additional_data=self._data
            )
            tasks.append(task)

        for task in asyncio.as_completed(tasks):
            result = None
            try:
                result = await task
            except Exception as e:
                msg = f"Unknown error in run_flows: {e}, {e.args=}"
                logger.exception(msg)
                # raise
            if result:
                if result.error:
                    logger.error(
                        f"Agent {agent.name} failed with error: {result.error}",
                    )
                else:
                    try:
                        self.incorporate_outputs(
                            step_name=agent.name,
                            result=result,
                            output_map=agent.outputs,
                        )
                    except Exception as e:
                        error_msg = (
                            f"Response data not formatted as expected: {e}, {e.args=}"
                        )
                        logger.error(error_msg)
                yield result
            await asyncio.sleep(0)

    def incorporate_outputs(self, step_name: str, result: Job, output_map: Mapping):
        """Update the data object with the outputs of the agent."""
        if result.error:
            # don't add failed jobs
            return
        elif output_map:
            output = parse_flow_vars(output_map, job=result, additional_data=self._data)
            for k,v in output.items():
                if isinstance(v, Sequence):
                    self._data[step_name][k] = self._data[step_name].get(k, []) 
                    self._data[step_name][k].extend(v)
                else:
                    self._data[step_name][k] = self._data[step_name].get(k, [])  
                    self._data[step_name][k].append(v)
        else:
            self._data[step_name]['outputs'] = self._data[step_name].get('outputs', [])  
            self._data[step_name]['outputs'].append(result.outputs.model_dump())

        pass
