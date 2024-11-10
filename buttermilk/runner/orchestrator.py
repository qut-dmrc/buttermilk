import asyncio
import random
from omegaconf import DictConfig, ListConfig, OmegaConf
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, field_serializer, model_validator
from typing import Coroutine, Mapping, Optional, Self, Sequence, AsyncGenerator, Tuple
from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.config import DataSource, Flow
from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.agents.lc import LC
from buttermilk.exceptions import FatalError
from buttermilk.runner.helpers import prepare_step_data
from buttermilk.utils.utils import expand_dict
from buttermilk.data.recordmaker import RecordMakerDF

class MultiFlowOrchestrator(BaseModel):
    data_generator: AsyncGenerator = Field(default_factory=RecordMakerDF)
    flow: Flow

    _num_runs: int = 1
    _concurrent: int = 20
    _tasks: Sequence[Coroutine] = PrivateAttr(default_factory=list)
    _dataset: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)


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

        # Prepare the data for the step
        # self._dataset = prepare_step_data(self.flow.data)
        return self

    async def make_tasks(self, data_generator, agent_type: type[Agent], source) -> AsyncGenerator[Tuple[str,str,Job], None]:
        # create and run a separate job for:
        #   * each record in the data_generator
        #   * each Agent (Different classes or instances of classes to resolve a task)

        # Get permutations of init variables
        agent_combinations = self.flow.agent
        flow = self.flow
        agent_combinations = expand_dict(agent_combinations)

        # Get permutations of run variables
        run_combinations = self.flow.parameters
        run_combinations = expand_dict(run_combinations)

        for init_vars in agent_combinations:
            agent = agent_type(flow=flow, **init_vars)
            async for record in data_generator():
                for run_vars in run_combinations:
                    job = Job(record=record, source=source, parameters=run_vars)
                    coroutine = agent.run(job)
                    yield agent.name, job.job_id, coroutine


    async def run_tasks(self) -> AsyncGenerator[Job, None]:
        _sem = asyncio.Semaphore(self._concurrent)

        async def task_wrapper(agent_name, job_id, task):
            try:
                async with _sem:
                    logger.debug(f"Starting task for Agent {agent_name} with job {job_id}.")
                    result = await task
                    self._tasks_remaining  -= 1

                if result.error:
                    logger.warning(f"Agent {agent_name} failed job {job_id} with error: {result.error}")
                    self._tasks_failed += 1
                else:
                    logger.debug(f"Agent {agent_name} completed job {job_id} successfully.")
                    self._tasks_completed += 1

                return result
            
            except Exception as e:
                raise FatalError(f"Task {agent_name} job: {job_id} failed with error: {e}, {e.args=}")

        for n in range(self._num_runs):
            async with asyncio.TaskGroup() as tg:
                async for agent_name, job_id, task in self.make_tasks(data_generator=self._data_generator, agent_type=self.agent_type, source=self.source):
                    wrapped_task = task_wrapper(agent_name, job_id, task)
                    tg.create_task(wrapped_task)
                    await asyncio.sleep(0)  # Yield control to allow tasks to start processing

                    
            # All tasks in this run are now complete
            logger.info(f"Completed run {n+1} of {self._num_runs}")

            
        # All runs are now complete
        logger.info("All tasks have completed.")