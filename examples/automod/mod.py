###
# You can debug from here, or run from the command line with:
#   python examples/automod/mod.py +experiments=trans +data=tja
###
import asyncio

import hydra
from tqdm.asyncio import tqdm as atqdm

from buttermilk.runner.batch import MultiFlowOrchestrator
from buttermilk.runner.runloop import run_tasks

# Run with, e.g.:
# ```bash
# python -m examples.automod.mod +data=drag +step=ordinary +save=bq job=<description> source=<desc>
# ```


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg) -> None:
    global bm, logger

    # Load the main ButterMilk singleton instance
    # This takes care of credentials, save paths, and other defaults
    bm = BM(cfg=cfg)
    logger = bm.logger

    async def run():
        _continue = True
        for step_cfg in cfg.flows:
            if _continue:

                # Create an orchestrator to conduct all combinations of jobs we want to run
                orchestrator = MultiFlowOrchestrator(flow=step_cfg, source=cfg.job)
                step_name = step_cfg.name

                with atqdm(
                    colour="magenta",
                    desc=step_name,
                ) as pbar:
                    main_task = asyncio.create_task(run_tasks(task_generator=orchestrator.make_tasks(), num_runs=step_cfg.num_runs, max_concurrency=cfg.run.max_concurrency))

                    while not main_task.done():
                        pbar.total = sum([orchestrator._tasks_remaining, orchestrator._tasks_completed, orchestrator._tasks_failed])
                        pbar.n = sum([orchestrator._tasks_completed, orchestrator._tasks_failed])
                        pbar.refresh()
                        await asyncio.sleep(1)

                # exiting here, check for clean exit
                # set _continue = False  if loop failed

    asyncio.run(run())


if __name__ == "__main__":
    main()
