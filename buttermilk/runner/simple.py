
from typing import Self

from pydantic import Field, PrivateAttr, model_validator

from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput
from buttermilk._core.orchestrator import Orchestrator


class Sequencer(Orchestrator):
    agents: list[list[Agent]] = Field(default_factory=list)
    _records: list = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def _register_agents(self) -> Self:
        for step in self.steps:
            step_agents = []
            for obj, cfg in step.get_configs():
                step_agents.append(obj(**cfg))
            self.agents.append(step_agents)
        return self

    async def run(self, request=None) -> None:
        prompt = await self.interface.get_input(message="Enter your prompt or record info...")
        for step in self.agents:
            for variant in step:
                mapped_inputs = self._flow_data._resolve_mappings(variant.inputs)

                step_inputs = AgentInput(prompt=prompt, inputs=mapped_inputs, records=self._records)
                result = await variant(step_inputs)
                await self.store_results(step=variant.name, result=result)

                await self.interface.send_output(result)

            prompt = ""
            if not await self.interface.confirm("Proceed to next step?"):
                break
        return

