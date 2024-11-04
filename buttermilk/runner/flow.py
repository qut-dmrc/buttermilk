# A flow takes a job, does something, and returns a job.

from functools import wraps
from typing import Any, Callable

from pydantic import BaseModel

from buttermilk import Job, Result
from buttermilk.runner.runner import ResultsCollector
from buttermilk.utils import save
from buttermilk.utils.save import upload_rows


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
        if not isinstance(response, Job):
            raise ValueError(f"Expected a Job to save, got: {type(response)} for: {response}")
        output = response.model_dump()
        try:
            # move metadata to the record id
            output['metadata']['output'] = output['outputs']['metadata']
            del output['outputs']['metadata']
        except (TypeError, KeyError):
            if 'outputs' not in output:
                output['outputs'] = {}
            pass

        return output

    def _save(self):
        try:
            if self.to_save:
                uri = upload_rows(
                    schema=self.dest_schema, rows=self.to_save, dataset=self.dataset
                )
                self.to_save = []  # Clear the batch after saving
        except Exception as e:
            # emergency save
            uri = save.save(self.to_save)
            raise e



async def run_flow(job: Job, flow: callable):
    response = await flow.call_async(**job.inputs)
    job.outputs = Result(**response)
    return job

def flow():
    def inner(func):
        @wraps(func)
        def _impl(job: Job) -> Job:
            return run_flow(job, func)
        return _impl
