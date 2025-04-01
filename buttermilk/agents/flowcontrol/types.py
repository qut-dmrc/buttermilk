from buttermilk._core.contract import ConductorRequest, ManagerMessage, ManagerRequest
from buttermilk.agents.llm import LLMAgent


class HostAgent(LLMAgent):
    """Coordinators for group chats that use an LLM."""

    async def handle_control_message(
        self,
        message: ManagerMessage | ManagerRequest | ConductorRequest,
    ) -> ManagerMessage | ManagerRequest:
        # Respond to a control question addressed to us
        return await self._process(message)
