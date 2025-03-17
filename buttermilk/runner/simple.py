
from pydantic import BaseModel

from buttermilk._core.contract import Orchestrator


class Sequencer(Orchestrator):

    async def run(self, job: Job) -> Job:
        results = job.inputs.model_copy()
        
        for step in self.flow:
            results[step.name] = await step(results)

        job.outputs = results
        return results
    
    