import asyncio
import atexit
import json
import signal
import time
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Self

from cloudpathlib import CloudPath
from promptflow.tracing import trace
from pydantic import BaseModel, ConfigDict, Field, model_validator

from buttermilk._core.runner_types import Job
from buttermilk.bm import BM, logger
from buttermilk.utils import save
from buttermilk.utils.save import upload_rows


################################
#
# Results collector
#
################################
class ResultsCollector(BaseModel):
    """A simple collector that receives results from a queue and collates them."""

    bm: BM = Field(default_factory=lambda: BM())
    results: asyncio.Queue[Job] = Field(default_factory=asyncio.Queue)
    shutdown: bool = False
    n_results: int = 0
    to_save: list = []
    batch_size: int = 50  # rows

    # # The URI or path to save all results from this run
    batch_path: CloudPath | Path = None

    @model_validator(mode="after")
    def get_path(self) -> Self:
        if self.batch_path is None:
            self.batch_path = CloudPath(self.bm.save_dir)
        return self

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def run(self):
        try:
            # Collect results
            while not self.shutdown or not self.results.empty():
                await asyncio.sleep(delay=1)
                try:
                    result = self.results.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(3.0)
                    continue

                # Add result to the batch
                row_data = self.process(result)
                self.to_save.append(row_data)
                self.n_results += 1

                if len(self.to_save) >= self.batch_size:
                    # Save the batch of results when batch size is reached
                    self.save_with_trace()

        except Exception as e:  # Log any errors that occur during result collection
            logger.error(f"Unable to collect results: {e} {e.args=}")
            raise e

        finally:
            self.shutdown = True
            self.save_with_trace()  # Save any remaining results in the batch

    # Turn a Job with results into a record dict to save
    def process(self, response: Job) -> dict[str, Any]:
        if isinstance(response, Job):
            return response.model_dump()
        return response

    @trace
    def save_with_trace(self, **kwargs):
        return self._save(**kwargs)

    def _save(self):
        if self.to_save:
            _save_path = self.batch_path / f"results_{self.n_results}.json"
            uri = self.bm.save(data=self.to_save, uri=_save_path.as_uri())
            self.to_save = []
        return uri


################################
#
# Results saver
#
################################
class ResultSaver(ResultsCollector):
    """Collects results from the output queue and saves them to BigQuery.

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
            output["metadata"]["output"] = output["outputs"]["metadata"]
            del output["outputs"]["metadata"]
        except (TypeError, KeyError):
            if "outputs" not in output:
                output["outputs"] = {}

        return output

    def _save(self):
        try:
            if self.to_save:
                uri = upload_rows(
                    schema=self.dest_schema,
                    rows=self.to_save,
                    dataset=self.dataset,
                )
                self.to_save = []  # Clear the batch after saving
        except Exception as e:
            # emergency save
            uri = save.save(self.to_save)
            raise e


class AsyncDataUploader:
    def __init__(
        self,
        upload_fn,
        buffer_size: int = 1000,
        flush_interval: int = 60,
    ):
        self.upload_fn = upload_fn
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self.queue = asyncio.Queue()
        self.buffer: list[Any] = []
        self.last_flush = time.time()
        self._shutdown = False

        self.backup_dir = Path(mkdtemp())

        # Start background worker
        self.worker_task = asyncio.create_task(self._worker())

        # Register shutdown handlers
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)

    async def add(self, item: Any):
        """Add item to upload queue."""
        await self.queue.put(item)
        await self._backup_item(item)

    async def _worker(self):
        """Background worker that processes the queue."""
        while not self._shutdown or not self.queue.empty():
            try:
                # Get item with timeout
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    self.buffer.append(item)
                except TimeoutError:
                    pass

                # Check if we should flush
                should_flush = (
                    len(self.buffer) >= self.buffer_size
                    or time.time() - self.last_flush >= self.flush_interval
                )

                if should_flush and self.buffer:
                    await self._flush()

            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)

    async def _flush(self):
        """Upload buffered items"""
        if not self.buffer:
            return

        try:
            await self.upload_fn(self.buffer)
            self.last_flush = time.time()
            self.buffer = []
            await self._clear_backup()
        except Exception as e:
            logger.error(f"Flush error: {e}")
            # Keep items in buffer for retry

    async def _backup_item(self, item):
        """Write item to backup file."""
        backup_file = self.backup_dir / f"backup_{datetime.now().isoformat()}.json"
        backup_file.write_text(json.dumps(item))

    async def _clear_backup(self):
        """Clear backup files after successful upload."""
        for f in self.backup_dir.glob("backup_*.json"):
            f.unlink()

    def shutdown(self, *args):
        """Graceful shutdown ensuring all data is flushed."""
        self._shutdown = True
        asyncio.run(self._flush())
