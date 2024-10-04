# A flow takes a job, does something, and returns a job.

from functools import wraps
from typing import Callable, Any

from pydantic import BaseModel

from buttermilk.runner.runner import ResultsCollector
from buttermilk.utils import save
from buttermilk.utils.save import upload_rows
from buttermilk.runner import Job



################################
#
# Results saver
#
################################
class ResultsSaver(ResultsCollector):
    """
    Collects results from the output queue and saves them to BigQuery.

    Continuously polls the results queue for processed data. When a batch size is reached,
    it calls the `save` method to store the results. It also handles data formatting and
    error logging during collection.
    """

    dataset: str
    dest_schema: Any

    # Turn a Job with results into a record dict to save
    def process(self, response: Job) -> dict[str, Any]:
        # Here we use the data field as the source of data to upload in each row
        if not isinstance(response.data, Job):
            raise ValueError(f"Expected a Job to save, got: {type(response)} for: {response}")
        return response.data.model_dump()

    def _save(self):
        try:
            if self.to_save:
                uri = upload_rows(
                    schema=self.dest_schema, rows=self.to_save, dataset=self.dataset
                )
                self.to_save = []  # Clear the batch after saving
        except Exception as e:
            # emergency save
            uri = save(self.to_save)
            raise e



async def run_flow(job: Job, flow: callable):
    inputs = {}
    data = job.record.model_dump()
    
    inputs = {k:data[k] for k in job.input_map.keys()}
    # for k, v in job.input_map.items():
    #     v = v.split(".")
    #     for part in v:
    #         last_value = getattr(data, part)
    #     inputs[k] = last_value
        
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
