import asyncio
import random
from pydantic import BaseModel, Field
from typing import Optional, Sequence, AsyncGenerator
from .._core.agent import Agent
from buttermilk.utils.utils import expand_dict
from .._core.runner_types import Job, RecordInfo

class MultiFlowOrchestrator(BaseModel):
    agents: Optional[Sequence[Agent]] = Field(default_factory=list)
    n_runs: int = 1
    steps: Optional[Sequence] = Field(default_factory=list)
    agent_vars: Optional[dict] = Field(default_factory=dict)
    init_vars: Optional[dict] = Field(default_factory=dict)
    run_vars: Optional[dict] = Field(default_factory=dict)

    async def make_agents(self, agent: type = Agent) -> AsyncGenerator[Agent, None]:
        # Get permutations of init variables
        for vars in expand_dict(self.init_vars):
            yield agent(**vars, **self.agent_vars)

    async def make_jobs(self, record: RecordInfo, source: str) -> AsyncGenerator[Job, None]:
        # Get permutations of run variables
        run_combinations = expand_dict(self.run_vars) * self.n_runs

        # Shuffle
        random.shuffle(run_combinations)

        # for vars in run_combinations:
        #     async for job in self.run_tasks(record, source):
        #         job = Job(record=record, source=INPUT_SOURCE, parameters=vars)
        #         yield job

    async def run_tasks(self, record: RecordInfo, source: str) -> AsyncGenerator[Job, None]:

        # For each agent, create tasks for each job
        workers = []
        for agent in self.agents:
                workers.append(agent.run(job))
        
        # Process tasks as they complete
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(worker) for worker in workers]
            
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    yield result
                except Exception as e:
                    bm.logger.error(f"Worker failed with error: {e}")
                    continue
        
        # All workers are now complete
        return
