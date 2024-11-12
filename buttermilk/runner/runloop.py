import asyncio
import datetime
import json
import os
import pickle
import sys
import time
import uuid
from abc import ABC, abstractmethod
from asyncio import Queue, QueueEmpty, Semaphore, TaskGroup, gather
from functools import cached_property, wraps
from pathlib import Path
from random import random
from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    List,
    Optional,
    Self,
    Sequence,
    Type,
    Union,
)

import numpy as np
import pandas as pd
import requests
import shortuuid
from cloudpathlib import CloudPath
from humanfriendly import format_timespan
from promptflow.tracing import start_trace, trace
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from tqdm.asyncio import tqdm as atqdm

from buttermilk.bm import BM
from buttermilk.exceptions import FatalError
from buttermilk._core.runner_types import Job
from buttermilk.utils.errors import extract_error_info
from buttermilk._core.log import logger

"""
A wrapper for the main execution loop, handling errors and interrupts (hopefully grafully)
"""


async def run_tasks(
    task_generator,
    num_runs=1,
    max_concurrency=32,
):
    """
    Starts the run, managing the consumers and collecting results.
    """

    _sem = Semaphore(max_concurrency)


    t0 = time.perf_counter()
    try:
        for n in range(num_runs):
            t1 = time.perf_counter()

            try:
                async with asyncio.TaskGroup() as tg:
                    async with _sem:
                        async for wrapped_task in task_generator:
                            tg.create_task(wrapped_task)
                            await asyncio.sleep(
                                0
                            )  # Yield control to allow tasks to start processing

            except KeyboardInterrupt:
                # we have been interrupted. Abort gracefully if possible -- the first time. The second time, abort immediately.
                logger.info(
                    "Keyboard interrupt. Finishing existing jobs and quitting. Interrupt again to quit immediately."
                )
            finally:
                # All tasks in this run are now complete
                time_taken = time.perf_counter() - t1
                logger.info(f"Completed run {n+1} of {num_runs} in {format_timespan(time_taken)}.")

                await asyncio.sleep(0)

    except KeyboardInterrupt:
        # Second time; quit immediately.
        raise FatalError("Keyboard interrupt. Aborting immediately.")
    except ExceptionGroup as eg:
        for e in eg.exceptions:
            logger.error(
                f"Received unhandled exception (in ExceptionGroup): {e}. Aborting.",
                extra={"traceback": e.__traceback__},
            )
        raise FatalError("Aborting run following exceptions in task group.")
    except Exception as e:
        logger.exception(f"Received unhandled exception in run loop! {e} {e.args=}")
        raise e

    finally:
        time_taken = time.perf_counter() - t0
        logger.info(f"Run finished in {format_timespan(time_taken)}.")
