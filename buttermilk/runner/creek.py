import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import (
    Any,
)

import hydra
from pydantic import BaseModel

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.config import SaveInfo
from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.bm import BM
from buttermilk.utils.utils import find_in_nested_dict

""" A little stream. Runs several flow stages over a single record and streams results."""


class Creek(BaseModel):
    source: str
    flows: list[Agent]

    _data: dict = {}

    class Config:
        arbitrary_types_allowed = True

    async def run_flows(self, record: RecordInfo) -> AsyncGenerator[Any, None]:
        save_data = SaveInfo(
            type="bq",
            dataset="dmrc-analysis.toxicity.flow",
            db_schema="buttermilk/schemas/flow.json",
        )

        for agent in self.flows:
            agent.save = save_data
            self._data[agent.name] = {}

            tasks = []
            for variant in agent.make_combinations():
                # Create a new job and task for every combination of variables
                # this agent is configured to run.
                job = Job(record=record, source=self.source)
                task = agent.process_job(job=job, additional_data=self._data, **variant)
                tasks.append(task)

            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    if result.error:
                        yield f"Error: {result.error}"
                    else:
                        try:
                            if agent.outputs:
                                for key, values in agent.outputs.items():
                                    if key not in self._data[agent.name]:
                                        self._data[agent.name][key] = []
                                    self._data[agent.name][key].append(
                                        self.extract(values, result),
                                    )
                        except Exception as e:
                            logger.exception(e)
                        yield result.outputs
                except Exception as e:
                    logger.exception(e)
                    raise

    def make_answers(self) -> list[dict[str, str]]:
        answers = []
        for rec in self._data:
            answer = {
                "id": rec.job_id,
                "model": rec.parameters["model"],
                "template": rec.agent_info.template,
                "reasons": rec.outputs.reasons,
            }
            answers.append(answer)
        return answers

    def extract(self, values: Any, result: Job):
        """Get data out of hierarchical results object according to outputs schema."""
        data = None
        if isinstance(values, str):
            data = find_in_nested_dict(result.model_dump(), values)
        elif isinstance(values, Sequence):
            data = [find_in_nested_dict(result.model_dump(), v) for v in values]
        elif isinstance(values, Mapping):
            data = {}
            for key, item in values.items():
                data[key] = find_in_nested_dict(result.model_dump(), item)

        return data


class Tester(BaseModel):
    name: str


class _CFG(BaseModel):
    bm: BM
    flows: Creek


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: _CFG) -> None:
    from rich import print as rprint

    # Hydra will automatically instantiate the objects
    # This should work, but doesn't?
    objs = hydra.utils.instantiate(cfg)
    creek = objs.flows

    text = """An image depicting a caricature of a Jewish man with an exaggerated hooked nose and a Star of David marked with "Jude" (resembling Holocaust-era badges), holding a music box labeled "media." A monkey labeled "BLM" sits on the man's shoulder."""

    record = RecordInfo(text=text)

    async def test_run_flow(creek: Creek):
        async for response in creek.run_flows(record=record):
            rprint(response)

    asyncio.run(test_run_flow(creek))


if __name__ == "__main__":
    main()
