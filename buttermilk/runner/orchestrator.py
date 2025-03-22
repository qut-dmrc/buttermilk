from collections.abc import AsyncGenerator, Coroutine, Sequence
from typing import Any

import shortuuid
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_serializer,
    model_validator,
)

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.flow import Flow
from buttermilk.data.recordmaker import RecordMakerDF
from buttermilk.exceptions import FatalError
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.utils.utils import expand_dict


class MultiFlowOrchestrator(BaseModel):
    flow: Flow
    source: str

    _num_runs: int = 1
    _concurrent: int = 20
    _tasks: Sequence[Coroutine] = PrivateAttr(default_factory=list)
    _dataset: Any = PrivateAttr(default=None)
    _data_generator: Any = PrivateAttr(default=None)

    _tasks_remaining: int = PrivateAttr(default=0)
    _tasks_completed: int = PrivateAttr(default=0)
    _tasks_failed: int = PrivateAttr(default=0)

    _agents: list[Agent] = PrivateAttr(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("flow")
    def serialize_omegaconf(cls, value):
        return OmegaConf.to_container(value, resolve=True)

    @model_validator(mode="after")
    def set_vars(self) -> "MultiFlowOrchestrator":
        self._num_runs = self.flow.num_runs or self._num_runs
        self._concurrent = self.flow.concurrency or self._concurrent
        return self

    async def prepare(self):
        # Prepare the data and agents for the step
        self._dataset = await prepare_step_df(self.flow.data)
        self._data_generator = RecordMakerDF(dataset=self._dataset).record_generator

        self._agents = [x async for x in self.make_agents()]

    async def make_agents(self):
        # Get permutations of init variables
        agent_combinations = expand_dict(self.flow.agent.model_dump())

        for init_vars in agent_combinations:
            if not init_vars.get("flow"):
                init_vars["flow"] = self.flow.name
            agent: Agent = globals()[self.flow.agent.type](
                **init_vars,
                save=self.flow.save,
            )
            yield agent

    async def make_tasks(self) -> AsyncGenerator[Coroutine, None]:
        # create and run a separate job for
        #   * each record in  self._data_generator
        #   * each Agent (Different classes or instances of classes to resolve a task)

        # Get permutations of run variables
        run_combinations = expand_dict(self.flow.parameters)

        async for record in self._data_generator():
            flow_id = shortuuid.uuid()
            for agent in self._agents:
                for run_vars in run_combinations:
                    job = Job(
                        flow_id=flow_id,
                        record=record,
                        source=self.source,
                        parameters=run_vars,
                    )
                    coroutine = agent.run(job)
                    coroutine = self.task_wrapper(
                        task=coroutine,
                        job_id=job.job_id,
                        agent_name=agent.id,
                    )
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
            raise FatalError(
                f"Task {agent_name} job: {job_id} failed with error: {e}, {e.args=}",
            )
