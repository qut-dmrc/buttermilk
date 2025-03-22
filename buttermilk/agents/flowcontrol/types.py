from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput, ManagerMessage


class HostAgent(Agent):
    """Special agent that can receive OOB requests."""

    async def handle_control_message(self, message: ManagerMessage) -> ManagerMessage:
        result = await self.process(
            input_data=AgentInput(
                content=message.content,
            ),
        )
        return ManagerMessage(**result.model_dump())
