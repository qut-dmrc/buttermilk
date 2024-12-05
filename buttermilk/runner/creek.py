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
from buttermilk.runner.helpers import prepare_step_df, prepare_flow_inputs
from buttermilk.utils.utils import find_in_nested_dict

""" A little stream. Runs several flow stages over a single record 
    and streams results.
"""


class Creek(BaseModel):
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
        self._data[agent.name] = []
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
            job = prepare_flow_inputs(job=job, additional_data=self._data, **agent.inputs, **variant)
            
            task = agent.run(
                job=job,
                q=q,
                additional_data=self._data,
                **variant,
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
        if output_map:
            output = {}
            for key, values in output_map.items():
                output[key] = self.extract(values, result)
            self._data[step_name].append(output)
        else:
            self._data[step_name].append(result.outputs.model_dump())

    def extract(self, values: Any, result: Job):
        """Get data out of hierarchical results object according to outputs schema."""
        data = None
        if isinstance(values, str):
            data = find_in_nested_dict(result.model_dump(), values)
        elif isinstance(values, Sequence):
            data = [find_in_nested_dict(result.model_dump(), v) for v in values]
        elif isinstance(values, Mapping):
            data = {}
            for key, item in values.items():
                data[key] = find_in_nested_dict(result.model_dump(), item)

        return data
