import asyncio
import random
from omegaconf import DictConfig, ListConfig, OmegaConf
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, field_serializer, model_validator
from typing import Coroutine, Mapping, Optional, Self, Sequence, AsyncGenerator, Tuple
from buttermilk import Agent, Job, RecordInfo, logger
from buttermilk._core.agent import SaveInfo
from buttermilk.agents.lc import LC
from buttermilk.runner.helpers import prepare_step_data
from buttermilk.utils.utils import expand_dict

class MultiFlowOrchestrator(BaseModel):
    
    save: SaveInfo
    step: DictConfig
    data: ListConfig

    _num_runs: int = 1
    _concurrent: int = 20
    _tasks: Sequence[Coroutine] = PrivateAttr(default_factory=list)
    _dataset: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)

    class Config:
        arbitrary_types_allowed = True

    @field_serializer('step', 'data')
    def serialize_omegaconf(cls, value):
        return OmegaConf.to_container(value, resolve=True)
    
    @model_validator(mode='after')
    def get_data(self) -> Self:
        self._num_runs = self.step.get('num_runs') or self._num_runs
        self._concurrent = self.step.get('concurrent') or self._concurrent

        # Prepare the data for the step
        self._dataset = prepare_step_data(data_config=self.data)
        return self

    async def make_tasks(self, data_generator, agent_type: type[Agent], source) -> AsyncGenerator[Tuple[str,str,Job], None]:
        # create and run a separate job for:
        #   * each record in the data_generator
        #   * each Agent (Different classes or instances of classes to resolve a task)

        # Get permutations of init variables
        agent_combinations = self.step.agent
        agent_name = self.step.get('name', agent_type.__name__)
        agent_combinations = expand_dict(agent_combinations)

        # Get permutations of run variables
        run_combinations = self.step.get('parameters', {})
        run_combinations = expand_dict(run_combinations)

        for init_vars in agent_combinations:
            agent = agent_type(**init_vars, save_params=self.save)
            async for record in data_generator():
                for run_vars in run_combinations:
                    job = Job(record=record, source=source, parameters=run_vars)
                    coroutine = agent.run(job)
                    yield agent_name, job.job_id, coroutine

    async def data_generator(self) -> AsyncGenerator[RecordInfo, None]:
        # Generator to yield records from the dataset
        for _, record in self._dataset.sample(frac=1).iterrows():
            yield RecordInfo(**record.to_dict())


    async def run_tasks(self) -> AsyncGenerator[Job, None]:
        _sem = asyncio.Semaphore(self._concurrent)

        async def task_wrapper(agent_name, job_id, task):
            try:
                async with _sem:
                    logger.info(f"Starting task for Agent {agent_name} with job {job_id}.")
                    result = await task

                if result.error:
                    logger.error(f"Agent {agent_name} failed job {job_id} with error: {result.error}")
                else:
                    logger.info(f"Agent {agent_name} completed job {job_id} successfully.")

                return result
            
            except Exception as e:
                logger.error(f"Task failed with error: {e}, {e.args=}")

        for n in range(self._num_runs):
            async with asyncio.TaskGroup() as tg:
                async for agent_name, job_id, task in self.make_tasks(data_generator=self.data_generator, agent_type=LC,source='batch'):
                    wrapped_task = task_wrapper(agent_name, job_id, task)
                    tg.create_task(wrapped_task)
                    
            # All tasks in this run are now complete
            logger.info(f"Completed run {n+1} of {self._num_runs}")

            
        # All runs are now complete
        logger.info("All tasks have completed.")