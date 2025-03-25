from pydantic import PrivateAttr
from buttermilk._core.contract import ManagerMessage, UserConfirm
from buttermilk.agents.llm import LLMAgent

class HostAgent(LLMAgent):
    """Coordinators for group chats that use an LLM."""


    async def handle_control_message(
        self,
        message: ManagerMessage | UserConfirm,
    ) -> ManagerMessage | UserConfirm:
        pass
        # raise NotImplementedError("HostAgent does not handle control messages yet...")
