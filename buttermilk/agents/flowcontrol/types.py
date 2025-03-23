from buttermilk._core.contract import ManagerMessage
from buttermilk.agents.llm import LLMAgent


class HostAgent(LLMAgent):
    """Coordinators for group chats that use an LLM."""

    async def handle_control_message(self, message: ManagerMessage) -> ManagerMessage:
        raise NotImplementedError("HostAgent does not handle control messages yet...")
