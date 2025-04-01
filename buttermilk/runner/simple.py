
from typing import Self

import shortuuid
from pydantic import PrivateAttr, model_validator

from buttermilk._core.agent import Agent
from buttermilk._core.contract import ManagerRequest, StepRequest
from buttermilk._core.orchestrator import Orchestrator
from buttermilk.agents.ui.generic import UIAgent


class Sequencer(Orchestrator):
    _agents: dict[str, list[Agent]] = PrivateAttr(default_factory=dict)
    _records: list = PrivateAttr(default_factory=list)
    _manager: UIAgent
    #  something like
    # _manager = CLIUserAgent(id=MANAGER, name="Manager",
    #   description="Console human interface", _input_task=None)

    @model_validator(mode="after")
    def _register_agents(self) -> Self:

        roles: list[dict[str, str]] = []
        for step_name, variant_cfg in self.agents.items():
            step_agents = []
            for agent_cls, variant in variant_cfg.get_configs():
                agent_cfg = variant.model_dump()
                agent_cfg["id"] = f"{step_name}-{shortuuid.uuid()[:6]}"

                step_agents.append(agent_cls(**agent_cfg))
            self._agents[step_name] = step_agents
            roles.append({"role": step_name, "description": variant_cfg.description})

        self._flow_data.add(key="participants", value=roles)
        return self

    async def run(self, request=None) -> None:
        if not (prompt := request):
            input_req = ManagerRequest(
                content="Enter your prompt or record info...",
            )
            prompt = await self._manager._request_user_input(message=input_req)

        for step_name, agent_list in self._agents.items():
            step_request = StepRequest(
                role=step_name,
                prompt=prompt,
                source=self.flow_name,
            )
            step_inputs = await self._prepare_step(step_request)
            for agent in agent_list:
                result = await agent(step_inputs)
                # await self.store_results(step=variant.name, result=result)

                await self._manager.receive_output(result)

            prompt = await self._manager._request_user_input(
                ManagerRequest(content="Proceed to next step?", confirm=True),
            )
            if not prompt:
                break
