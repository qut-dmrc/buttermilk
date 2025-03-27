from buttermilk._core.contract import ManagerMessage, UserRequest
from buttermilk.agents.llm import LLMAgent


class HostAgent(LLMAgent):
    """Coordinators for group chats that use an LLM."""

    async def handle_control_message(
        self,
        message: ManagerMessage | UserRequest,
    ) -> ManagerMessage | UserRequest:
        pass
        # raise NotImplementedError("HostAgent does not handle control messages yet...")
