import asyncio
from functools import partial
import random
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, field_serializer, model_validator
from typing import Any, Coroutine, Mapping, Optional, Self, Sequence, AsyncGenerator, Tuple
from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentInfo, DataSource, Flow, Project, SaveInfo
from buttermilk._core.runner_types import Job, RecordInfo
from buttermilk.agents.lc import LC
from buttermilk.buttermilk import BM
from buttermilk.exceptions import FatalError
from buttermilk.runner.helpers import prepare_step_df
from buttermilk.runner.orchestrator import MultiFlowOrchestrator
from buttermilk.utils.utils import expand_dict
from buttermilk.data.recordmaker import RecordMaker, RecordMakerDF

""" A little stream. Runs several flow stages over a single record and streams results."""
class Creek(BaseModel):
    # flows: dict[str, Sequence[Flow]]
    source: str
    
    _data: list = []

    class Config:
        arbitrary_types_allowed = True

    async def record_generator(self, record: RecordInfo) -> AsyncGenerator[RecordInfo, None]:
        # Just yield one record
        yield record

    async def run_flow(self, record: RecordInfo, flow: Flow) -> AsyncGenerator[str, None]:
        orchestrator = MultiFlowOrchestrator(flow=flow, source=self.source)
        await orchestrator.make_agents()
        orchestrator._data_generator = partial(self.record_generator, record=record)
        tasks = [x async for x in orchestrator.make_tasks()]
        for result in asyncio.as_completed(tasks):
            try:
                result = await result
                if result.error:
                    yield f"Error: {result.error}"
                else:
                    self._data.append(result)
                    yield result.outputs
            except Exception as e:
                yield f"Error: {e}"

    async def run(self, record: RecordInfo):
        lc_judge = AgentInfo(type="LC", name="judger", template="judge", criteria="simplified", formatting="json_rules", model=["gpt4o", "sonnet"])
        save_data = SaveInfo(type="bq", dataset= "dmrc-analysis.toxicity.flow", db_schema="buttermilk/schemas/flow.json")

        judger = Flow(name="judger", concurrency=20, agent=lc_judge, save=save_data,parameters={"content": record.text})
        async for result in self.run_flow(flow=judger, record=record):
            yield result

        answers = self.make_answers()
        lc_synth = AgentInfo(type="LC", name="synth", template="synthesise", criteria="simplified", formatting="json_rules", model=["sonnet"])
        synth = Flow(name="synth", concurrency=20, agent=lc_synth, save=save_data)
        async for result in self.run_flow(flow=synth, record=record):
            yield result
        
    def make_answers(self) -> list[dict[str,str]]:
        answers = []
        for rec in self._data:
            answer = dict(
                id=rec.job_id,
                model=rec.parameters['model'],
                template=rec.agent_info.template,
                reasons=rec.outputs.reasons,
            )
            answers.append(answer)
        return answers

@hydra.main(version_base="1.3", config_path="../../examples/conf", config_name="config")
def main(cfg: Project) -> None:
    from rich import print as rprint
    bm = BM(cfg=cfg)
    creek = Creek(source="test")
    text =  """An image depicting a caricature of a Jewish man with an exaggerated hooked nose and a Star of David marked with "Jude" (resembling Holocaust-era badges), holding a music box labeled "media." A monkey labeled "BLM" sits on the man's shoulder."""

    async def test_run_flow(creek):
        async for response in creek.run(record=RecordInfo(content=text)):
            rprint(response)
            pass

    asyncio.run(test_run_flow(creek))


if __name__ == '__main__':
    main()