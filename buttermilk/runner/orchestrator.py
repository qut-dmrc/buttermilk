import asyncio
import random
from omegaconf import DictConfig, ListConfig, OmegaConf
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, field_serializer, model_validator
from typing import Any, Coroutine, Mapping, Optional, Self, Sequence, AsyncGenerator, Tuple
from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.config import DataSource, Flow
from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.agents.lc import LC
from buttermilk.exceptions import FatalError
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.utils.utils import expand_dict
from buttermilk.data.recordmaker import RecordMaker, RecordMakerDF

class MultiFlowOrchestrator(BaseModel):
    flow: Flow
    source: str

    _num_runs: int = 1
    _concurrent: int = 20
    _tasks: Sequence[Coroutine] = PrivateAttr(default_factory=list)
    _dataset: Any = PrivateAttr(default_factory=None)
    _data_generator: Any = PrivateAttr(default=None)

    _tasks_remaining: int = PrivateAttr(default=0)
    _tasks_completed: int = PrivateAttr(default=0)
    _tasks_failed: int = PrivateAttr(default=0)

    class Config:
        arbitrary_types_allowed = True

    @field_serializer('flow')
    def serialize_omegaconf(cls, value):
        return OmegaConf.to_container(value, resolve=True)
    
    @model_validator(mode='after')
    def get_data(self) -> Self:
        self._num_runs = self.flow.num_runs or self._num_runs
        self._concurrent = self.flow.concurrency or self._concurrent
        return self

    async def prepare_data(self):
        # Prepare the data for the step
        self._dataset = await prepare_step_df(self.flow.data)
        self._data_generator = RecordMakerDF(dataset=self._dataset).record_generator

    async def make_tasks(self) -> AsyncGenerator[Coroutine, None]:
        # create and run a separate job for:
        #   * each record in  self._data_generator
        #   * each Agent (Different classes or instances of classes to resolve a task)
        
        await self.prepare_data()

        # Get permutations of init variables
        agent_combinations = expand_dict(self.flow.agent.model_dump())

        # Get permutations of run variables
        run_combinations = expand_dict(self.flow.parameters)

        for init_vars in agent_combinations:
            if not init_vars.get('flow'):
                init_vars['flow'] = self.flow.name
            agent: Agent = globals()[self.flow.agent.type](**init_vars, save=self.flow.save)
            async for record in self._data_generator():
                for run_vars in run_combinations:
                    job = Job(record=record, source=self.source, parameters=run_vars)
                    coroutine = agent.run(job)
                    coroutine = self.task_wrapper(task=coroutine, job_id=job.job_id, agent_name=agent.name)
                    yield coroutine


    async def task_wrapper(self, *, agent_name, job_id, task):
        try:
            logger.debug(f"Starting task for Agent {agent_name} with job {job_id}.")
            result = await task
            self._tasks_remaining -= 1

            if result.error:
                logger.warning(f"Agent {agent_name} failed job {job_id} with error: {result.error}")
                self._tasks_failed += 1
            else:
                logger.debug(f"Agent {agent_name} completed job {job_id} successfully.")
                self._tasks_completed += 1

            return result
        
        except Exception as e:
            raise FatalError(f"Task {agent_name} job: {job_id} failed with error: {e}, {e.args=}")

