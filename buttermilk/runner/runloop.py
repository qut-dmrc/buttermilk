import asyncio
import time
from asyncio import Semaphore

from humanfriendly import format_timespan

from buttermilk import FatalError, logger

"""
A wrapper for the main execution loop, handling errors and interrupts (hopefully gracefully)
"""


async def run_tasks(
    task_generator,
    num_runs=1,
    max_concurrency=32,
):
    """Starts the run, managing the consumers and collecting results."""
    _sem = Semaphore(max_concurrency)

    t0 = time.perf_counter()
    try:
        for n in range(num_runs):
            t1 = time.perf_counter()

            try:
                async with asyncio.TaskGroup() as tg, _sem:
                    async for wrapped_task in task_generator:
                        tg.create_task(wrapped_task)
                        await asyncio.sleep(
                            0,
                        )  # Yield control to allow tasks to start processing

            except KeyboardInterrupt:
                # we have been interrupted. Abort gracefully if possible -- the first time. The second time, abort immediately.
                logger.info(
                    "Keyboard interrupt. Finishing existing jobs and quitting. Interrupt again to quit immediately.",
                )
            finally:
                # All tasks in this run are now complete
                time_taken = time.perf_counter() - t1
                logger.info(
                    "Completed run",
                    run_num=n + 1,
                    total_runs=num_runs,
                    time_taken=format_timespan(time_taken),
                )

                await asyncio.sleep(0)

    except KeyboardInterrupt:
        # Second time; quit immediately.
        raise FatalError("Keyboard interrupt. Aborting immediately.")
    except ExceptionGroup as eg:
        for e in eg.exceptions:
            logger.error(
                "Received unhandled exception (in ExceptionGroup)",
                error=e,
                extra={"traceback": e.__traceback__},
            )
        raise FatalError("Aborting run following exceptions in task group.")
    except Exception as e:
        logger.exception("Received unhandled exception in run loop!", error=e, error_args=e.args)
        raise

    finally:
        time_taken = time.perf_counter() - t0
        logger.info("Run finished", time_taken=format_timespan(time_taken))
