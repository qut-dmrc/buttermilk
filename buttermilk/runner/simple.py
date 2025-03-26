
from typing import Self

from pydantic import PrivateAttr, model_validator

from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput
from buttermilk._core.orchestrator import Orchestrator


class Sequencer(Orchestrator):
    _agents: list[list[Agent]] = PrivateAttr(default_factory=list)
    _records: list = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def _register_agents(self) -> Self:
        roles: list[dict[str, str]] = []
        for step_name, step in self.agents.items():
            step_agents = []
            for obj, cfg in step.get_configs():
                step_agents.append(obj(**cfg))
            self._agents.append(step_agents)
            roles.append({"role": step_name, "description": step.description})

        self._flow_data.add(key="participants", value=roles)
        return self

    async def run(self, request=None) -> None:
        prompt = await self._interface.get_input(
            message="Enter your prompt or record info...",
        )
        for step_name, step in self.agents.items():
            for variant in step:
                mapped_inputs = self._flow_data._resolve_mappings(variant.inputs)

                step_inputs = AgentInput(
                    agent_id=step_name,
                    content=prompt,
                    inputs=mapped_inputs,
                    records=self._records,
                )
                result = await variant(step_inputs)
                await self.store_results(step=variant.name, result=result)

                await self._interface.send_output(result)

            prompt = ""
            if not await self._interface.confirm("Proceed to next step?"):
                break
