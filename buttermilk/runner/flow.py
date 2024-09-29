# A flow takes a job, does something, and returns a job.

from typing import Callable


class Flow(BaseModel):
    func: Callable[[Job], Job]
    input_map: dict = {}

    def __call__(self, job: Job, data) -> Job:
        job.inputs = map(self.input_map.keys(), data[self.input_map.values()])
        result = self.func(**job.inputs)
        job.result = result

        return job
