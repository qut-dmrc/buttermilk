from buttermilk._core.contract import AgentInput, ManagerMessage
from buttermilk.agents.llm import LLMAgent


class HostAgent(LLMAgent):
    """Coordinators for group chats that use an LLM."""

    async def handle_control_message(self, message: ManagerMessage) -> ManagerMessage:
        result = await self.process(
            input_data=AgentInput(
                content=message.content,
            ),
        )
        return ManagerMessage(**result.model_dump())
