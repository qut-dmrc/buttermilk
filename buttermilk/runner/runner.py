"""
This module provides a robust framework for collecting and/or processing data in an asynchronous manner. It uses a producer-consumer model, where multiple Consumer instances (which must be subclassed) handle data processing, and the Distributor manages the overall execution, result collection, and storage. This design is ideal for large-scale, parallelizable data tasks.

Core Classes:

    Consumer (Abstract Base Class):
        Purpose: Each consumer takes data from an input queue and processes it. This class should be subclassed to implement your custom data processing logic.
        Key Attributes:
            input_queue: Queue for receiving data to process.
            output_queue: Queue for sending processed results.
            worker_name: Identifier for the worker.
            task_name: Description of the task being performed.
        Abstract Method:
            process(self, record: Any): Must be implemented by subclasses to define the actual data processing logic. This method is an asynchronous generator, yielding processed data as it becomes available.

    Distributor:
        Purpose: Orchestrates the data processing pipeline.
        Key Attributes:
            results: Queue for storing processed results.
            consumers: List of Consumer instances for data processing.
            data: Pandas DataFrame for tracking processing progress.
            name: Identifier for the runner.
            job: Name or description of the job being executed.
        Key Methods:
            batch(self, n_passes: int, examples: pd.DataFrame): Initiates a data processing batch with multiple passes over the input examples.
            prepare(self, examples: pd.DataFrame, n_passes: int = 1): Prepares the runner by populating input queues for the Consumer instances.
            run(self): Starts the asynchronous execution of the workers and results collection.
            collect(self): Gathers results from the results queue and saves them in batches.
            cleanup(self): Handles cleanup tasks after execution, including saving results to Google Sheets or Google Cloud Storage.
            save(self, batch=Optional[List[tuple]]): Abstract method (to be implemented in subclasses) for saving the results.

Key Features:

    Asynchronous Execution: Leverages asyncio for efficient concurrent processing.
    Progress Tracking: Uses tqdm to provide visual progress updates.
    Data Storage: Supports saving results to Google Sheets, Google Cloud Storage, or other formats (via subclass implementation).
    Fault Tolerance: Attempts to collect and save remaining results if an error occurs.
    Logging: Utilizes the datatools.log module for logging messages and errors.

Usage:
    Input data: You should provide an iterable or generator of input records. Each record should have a unique index.

    Subclass Consumer: Create your own worker for each separate task you wish to run on each input record. Each worker should inherit from Consumer and implement the process method. Each distinct task must have a unique name. You may create multiple workers with the same Task name to enable parallel (asynchronous) processing.

    Instantiate Distributor:
        * Create a Distributor instance, providing it with the list of your Consumer objects.
        * To upload results to BigQuery, also provide a full table identifier (`project.dataset.table`) and accompanying JSON schema.

    Run the batch Method: Call batch on your Distributor instance, passing in the input data and any necessary configuration parameters.

Example:
    ```Python

    ```
"""

import asyncio
import datetime
import json
import os
import pickle
import sys
import time
import uuid
from abc import ABC, abstractmethod
from asyncio import Queue, QueueEmpty, TaskGroup, gather
from functools import cached_property, wraps
from pathlib import Path
from random import random
from typing import Any, AsyncGenerator, Coroutine, List, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd
import requests
import shortuuid
from cloudpathlib import CloudPath
from humanfriendly import format_timespan
from promptflow.tracing import start_trace, trace
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from tqdm.asyncio import tqdm as atqdm


from buttermilk.runner._runner_types import Job
from buttermilk.utils.errors import extract_error_info

# The schema for the "runs" table in a BigQuery dataset
RUNS_SCHEMA = """
[{"name":"runtime","type":"TIMESTAMP","mode":"REQUIRED"},{"name":"name","type":"STRING","mode":"REQUIRED"},{"name":"arguments","type":"STRING","mode":"NULLABLE"},{"name":"task","type":"STRING","mode":"NULLABLE"},{"name":"meta","type":"STRING","mode":"NULLABLE","description":"A serialised representation of a dict with extra fields"},{"name":"successful","type":"BOOLEAN","mode":"NULLABLE"}]
"""
RUNS_SCHEMA = json.loads(RUNS_SCHEMA)

# The full path to the "runs" table in a BigQuery dataset
RUNS_TABLE = "dmrc-platforms.scrapers.runs"

runner_ip: str | None = None


################################
#
# Consumer with task definition and runner
#
################################
class Consumer(BaseModel):
    """
    Abstract base class for data processing workers (consumers).

    Each consumer takes data from an input queue, processes it, and sends the results to an output queue.

    Subclasses MUST implement the `process` method to define their specific data processing logic.

    Attributes:
        task_name (str): The name of the task.
        input_queue (Queue): The input queue for receiving data to process.
        output_queue (Queue): The output queue for sending processed results.
        task_num (int): The task number for this worker.
        concurrent (int): The number of concurrent tasks this worker can handle.
        done (bool): Flag to indicate whether the worker has finished or not.

    Methods:
        process(record: Job) -> AsyncGenerator[Job, None]: Process the data (to be implemented by subclasses). This method may add recursive tasks by yielding additional records to be processed by another step.
        run(self) -> None: Run the worker asynchronously until finished or told to stop.
    """

    task_name: str
    input_queue: Queue[Job] = Field(default_factory=Queue)
    output_queue: Queue[Job] = None
    task_num: Optional[int] = None
    concurrent: int = 1  # Number of async tasks to run

    # flag to stop the task or to indicate the queue has finished
    done: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Worker name for identification and logging
    @computed_field
    @cached_property
    def worker_name(self) -> str:
        return f"{self.task_name}_{shortuuid.uuid()[:6]}"

    @model_validator(mode="after")
    def validate_concurrent(self):
        if self.concurrent < 1:
            raise ValueError("concurrent must be at least 1")

        self._sem = asyncio.Semaphore(value=self.concurrent)
        return self

    @computed_field
    @cached_property
    def pbar(self) -> atqdm:
        colours = ["yellow", "green", "cyan", "blue", "magenta"]
        if self.task_num:
            colour = colours[self.task_num % len(colours)]
        else:
            colour = None
        return atqdm(
            total=self.input_queue.qsize(),
            dynamic_ncols=True,
            position=self.task_num,
            desc=self.task_name,
            bar_format="{desc:20}: {bar:50} | {rate_inv_fmt}",
            colour=colour,
        )

    async def run(self):
        try:
            async with asyncio.TaskGroup() as tg:
                while not self.done:
                    await asyncio.sleep(delay=0.1)
                    active_tasks = sum(1 for task in tg._tasks if not task.done())
                    if active_tasks >= self.concurrent * 2:
                        continue

                    # Schedule the job for processing, sending
                    # the result to self.output_queue
                    tg.create_task(self.process_wrapper())

        except Exception as e:
            # If we hit here, all remaining tasks for this worker will be canceled.
            logger.error(
                f"Canceling worker {self.worker_name} and remaining tasks after hitting error: {e} {e.args=}"
            )

        finally:
            self.done = True
            self.pbar.close()

    async def process_wrapper(self):
        """
        Wrapper for the process method.

        This method is used to catch and log any errors that occur during processing.
        """
        try:
            await asyncio.sleep(delay=0.1)
            # only run self.concurrent tasks at a time
            async with self._sem:
                # get a task from the queue
                try:
                    job = self.input_queue.get_nowait()
                except QueueEmpty:
                    self.done = True
                    return

                try:
                    async for result in self.process_trace(job=job):
                        await self.output_queue.put(result)
                        pass

                except Exception as e:
                    job.error = extract_error_info(e=e)
                    await self.output_queue.put(job)

                # mark input job done
                self.input_queue.task_done()
                self.pbar.update(1)
                self.pbar.refresh()
        except Exception as e:
            logger.error(
                f"Error processing task {self.task_name} by {self.worker_name} with job {job.job_id}. Error: {e or type(e)} {e.args=}"
            )

    @trace
    async def process_trace(self, *, job: Job) -> AsyncGenerator[Job, Any]:
        # This allows a worker to yield multiple results from a single input
        async for result in self.process(job=job):
            yield result
            await asyncio.sleep(delay=0.1)

    @abstractmethod
    async def process(self, *, job: Job) -> AsyncGenerator[Job, Any]:
        """
        Abstract method for data processing.

        This method MUST be implemented by subclasses. It should take a data record,
        process it, and return the processed result.
        """
        yield job


################################
#
# Results collector
#
################################
class ResultsCollector(BaseModel):
    """A simple collector that receives results from a queue and collates them."""

    results: Queue[Job] = Field(default_factory=Queue)
    shutdown: bool = False
    n_results: int = 0
    to_save: list = []
    batch_size: int = 50  # rows

    # The URI or path to save all results from this run
    batch_path: Union[CloudPath, Path] = Field(
        default_factory=lambda: CloudPath(gc.save_dir)
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @cached_property
    def pbar(self) -> atqdm:
        return atqdm(
            total=self.batch_size,
            dynamic_ncols=False,
            desc="Save buffer",
            bar_format="{desc:20}: {bar:20}",
            colour="#6610f2",  # (indigo)
            position=1,
        )

    async def run(self):
        try:
            # Collect results
            while not self.shutdown or not self.results.empty():
                await asyncio.sleep(delay=1)
                try:
                    result = self.results.get_nowait()
                    self.pbar.update(1)
                    self.pbar.refresh()
                except QueueEmpty:
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
        # By default, just dump the result
        return response.model_dump()

    @trace
    def save_with_trace(self, **kwargs):
        return self._save(**kwargs)

    def _save(self):
        _save_path = self.batch_path / f"results_{self.n_results}.json"
        gc.save(data=self.to_save, uri=_save_path.as_uri())
        self.to_save = []

################################
#
# Distributor class
#
################################
class TaskDistributor(BaseModel):
    """
    Adds tasks to the Queue. Maintains several queues at a time, identified by a task name.

    Accepts results on behalf of collectors.

    Future functionality: will also receive intermediate results and route them for more
    processing as required.
    """

    _consumers: dict[str, Consumer] = {}
    shutdown: bool = False
    _tasks: List[Coroutine] = []
    total_tasks: int = 0
    _collector: Optional[ResultsCollector] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def register_collector(
        self, collector: Optional[ResultsCollector] = None, /, **kwargs
    ):
        """Add a results saving worker and its input/outqut queues."""
        if not collector:
            collector = ResultsCollector(**kwargs)

        self._collector = collector

    def register_task(self, consumer: Consumer):
        """Add a consumer worker and its input/outqut queues."""
        if not self._collector:
            raise ValueError("Collector not registered. Do that first.")

        if consumer.task_name in self._consumers.keys():
            raise ValueError(f"Task {consumer.task_name} already exists")

        # set position for progressbar, making room for global and save rows
        consumer.task_num = len(self._consumers.keys()) + 2

        if consumer.output_queue is None:
            # By default, attach the main results queue to the consumer
            consumer.output_queue = self._collector.results

        self._consumers[consumer.task_name] = consumer

    def add_job(self, task_name: str, job: Job):
        """Add a task to the corresponding queue."""

        self._consumers[task_name].input_queue.put_nowait(job)
        self.total_tasks += 1

    async def run(self):
        """
        Starts the run, managing the consumers and collecting results.
        """

        t0 = time.perf_counter()
        try:
            # Set up tracing using promptflow
            start_trace(collection=bm.name, resource_attributes=dict(job=bm.job))

            global_pbar = atqdm(
                total=self.total_tasks,
                dynamic_ncols=False,
                position=0,
                # lock_args=(False,),
                desc="overall",
                colour="red",
                bar_format="{desc:20}: {bar:30} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )

            # Add the results saving task and initialise the results queue first
            self._collector_task = asyncio.create_task(self._collector.run())

            try:
                # And keep the workers separate
                async with TaskGroup() as tg:
                    for w in self._consumers.values():
                        # Create a task for each worker
                        self._tasks.append(tg.create_task(w.run()))

                    while (
                        not self.shutdown                   # Global shutdown flag not set
                        and not self._collector.shutdown    # Collector has not stopped
                        and len(self._consumers) > 0        # Consumers still alive
                        and not all([w.done for w in self._consumers.values()])  # Consumers still working
                    ):
                        await asyncio.sleep(0.1)
                        for w in self._consumers.values():
                            await asyncio.sleep(0.1)
                            w.pbar.refresh()
                            if w.done and not w.input_queue.empty():
                                # at least one of our consumers has died prematurely. We'll keep going with the others though.
                                logger.warning(
                                    f"Consumer {w.worker_name} died with {w.input_queue.qsize()} items left in the queue of type `{w.task_name}. Continuing other tasks."
                                )
                                continue
                        self._collector.pbar.refresh()
                        # update progress
                        global_pbar.update(self._collector.n_results - global_pbar.n)
                        global_pbar.refresh()

            except KeyboardInterrupt:
                # we have been interrupted. Abort gracefully if possible -- the first time, just stop any consumers getting new jobs and wait for them. The second time, abort immediately.
                logger.info(
                    "Keyboard interrupt. Finishing existing jobs and quitting. Interrupt again to quit immediately."
                )

            self.shutdown = True
            for w in self._consumers.values():
                w.done = True

            global_pbar.close()

            await asyncio.sleep(0.1)

            self._collector.shutdown = True
            await self._collector_task

        except KeyboardInterrupt:
            # Second time; quit immediately.
            raise FatalError("Keyboard interrupt. Aborting immediately.")
        except ExceptionGroup as eg:
            for e in eg.exceptions:
                logger.error(
                    f"Received unhandled exception (in ExceptionGroup): {e}. Aborting.",
                    extra={"traceback": e.__traceback__},
                )
        except Exception as e:
            logger.exception(
                f"Received unhandled exception! {e}",
                extra={"traceback": e.__traceback__, "args": e.args},
            )
        finally:
            time_taken = time.perf_counter() - t0
            logger.info(f"Run finished in {format_timespan(time_taken)}.")


