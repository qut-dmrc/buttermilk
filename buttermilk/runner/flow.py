# A flow takes a job, does something, and returns a job.

from functools import wraps
from typing import Callable

from pydantic import BaseModel

from buttermilk.runner import Job


async def run_flow(job: Job, flow: callable):
    data = job.record.model_dump()
    inputs = {k: data[v] for k, v in job.input_map.items()}
    try:
        job.outputs = await flow.call_async(**inputs)
    except Exception as e:
        job.error = str(e)
    return job

def flow():
    def inner(func):
        @wraps(func)
        def _impl(job: Job) -> Job:
            return run_flow(job, func)
        return _impl
