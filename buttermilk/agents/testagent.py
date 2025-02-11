import json
import random

from buttermilk._core.agent import Agent
from buttermilk._core.config import SaveInfo
from buttermilk._core.runner_types import Job

with open("buttermilk/api/test_data/test_jobs.json") as f:
    jobs = json.load(f)


class TestAgent(Agent):
    name: str = "test agent"
    save: SaveInfo | None = None

    async def process_job(
        self,
        *,
        job: Job,
        **kwargs,
    ) -> Job:
        job.outputs = random.choice(jobs)
        return job
